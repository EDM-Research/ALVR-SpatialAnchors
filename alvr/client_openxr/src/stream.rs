use crate::{
    from_xr_pose,
    graphics::{self, CompositionLayerBuilder},
    interaction::{self, InteractionContext},
    to_xr_fov, to_xr_pose, XrContext,
};
use alvr_client_core::{
    graphics::{GraphicsContext, StreamRenderer},
    ClientCoreContext, DecodedFrame, Platform,
};
use alvr_common::{
    error,
    glam::{self, UVec2, Vec2, Vec3},
    Pose, RelaxedAtomic, HAND_LEFT_ID, HAND_RIGHT_ID,
};
use alvr_packets::{ButtonValue, FaceData, StreamConfig, ViewParams};
use alvr_session::{
    BodyTrackingSourcesConfig, ClientsideFoveationConfig, ClientsideFoveationMode, EncoderConfig,
    FaceTrackingSourcesConfig, FoveatedEncodingConfig,
};
use openxr::{self as xr, sys, ViewConfigurationType};
use std::{
    rc::Rc,
    sync::Arc,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

// When the latency goes too high, if prediction offset is not capped tracking poll will fail.
const MAX_PREDICTION: Duration = Duration::from_millis(70);

#[derive(PartialEq)]
pub struct ParsedStreamConfig {
    pub view_resolution: UVec2,
    pub refresh_rate_hint: f32,
    pub foveated_encoding_config: Option<FoveatedEncodingConfig>,
    pub clientside_foveation_config: Option<ClientsideFoveationConfig>,
    pub encoder_config: EncoderConfig,
    pub face_sources_config: Option<FaceTrackingSourcesConfig>,
    pub body_sources_config: Option<BodyTrackingSourcesConfig>,
    pub prefers_multimodal_input: bool,
}

impl ParsedStreamConfig {
    pub fn new(config: &StreamConfig) -> ParsedStreamConfig {
        ParsedStreamConfig {
            view_resolution: config.negotiated_config.view_resolution,
            refresh_rate_hint: config.negotiated_config.refresh_rate_hint,
            foveated_encoding_config: config
                .negotiated_config
                .enable_foveated_encoding
                .then(|| config.settings.video.foveated_encoding.as_option().cloned())
                .flatten(),
            clientside_foveation_config: config
                .settings
                .video
                .clientside_foveation
                .as_option()
                .cloned(),
            encoder_config: config.settings.video.encoder_config.clone(),
            face_sources_config: config
                .settings
                .headset
                .face_tracking
                .as_option()
                .map(|c| c.sources.clone()),
            body_sources_config: config
                .settings
                .headset
                .body_tracking
                .as_option()
                .map(|c| c.sources.clone()),
            prefers_multimodal_input: config
                .settings
                .headset
                .controllers
                .as_option()
                .map(|c| c.multimodal_tracking)
                .unwrap_or(false),
        }
    }
}

pub struct StreamContext {
    core_context: Arc<ClientCoreContext>,
    xr_context: XrContext,
    interaction_context: Arc<InteractionContext>,
    reference_space: Arc<xr::Space>,
    swapchains: [xr::Swapchain<xr::OpenGlEs>; 2],
    view_resolution: UVec2,
    refresh_rate: f32,
    last_good_view_params: [ViewParams; 2],
    input_thread: Option<JoinHandle<()>>,
    input_thread_running: Arc<RelaxedAtomic>,
    renderer: StreamRenderer,
}

impl StreamContext {
    pub fn new(
        core_ctx: Arc<ClientCoreContext>,
        xr_ctx: XrContext,
        gfx_ctx: Rc<GraphicsContext>,
        interaction_ctx: Arc<InteractionContext>,
        platform: Platform,
        config: &ParsedStreamConfig,
    ) -> StreamContext {
        if xr_ctx.instance.exts().fb_display_refresh_rate.is_some() {
            xr_ctx
                .session
                .request_display_refresh_rate(config.refresh_rate_hint)
                .unwrap();
        }

        let foveation_profile = if let Some(config) = &config.clientside_foveation_config {
            if xr_ctx.instance.exts().fb_swapchain_update_state.is_some()
                && xr_ctx.instance.exts().fb_foveation.is_some()
                && xr_ctx.instance.exts().fb_foveation_configuration.is_some()
            {
                let level;
                let dynamic;
                match config.mode {
                    ClientsideFoveationMode::Static { level: lvl } => {
                        level = lvl;
                        dynamic = false;
                    }
                    ClientsideFoveationMode::Dynamic { max_level } => {
                        level = max_level;
                        dynamic = true;
                    }
                };

                xr_ctx
                    .session
                    .create_foveation_profile(Some(xr::FoveationLevelProfile {
                        level: xr::FoveationLevelFB::from_raw(level as i32),
                        vertical_offset: config.vertical_offset_deg,
                        dynamic: xr::FoveationDynamicFB::from_raw(dynamic as i32),
                    }))
                    .ok()
            } else {
                None
            }
        } else {
            None
        };

        let format =
            graphics::swapchain_format(&gfx_ctx, &xr_ctx.session, config.encoder_config.enable_hdr);

        let swapchains = [
            graphics::create_swapchain(
                &xr_ctx.session,
                &gfx_ctx,
                config.view_resolution,
                format,
                foveation_profile.as_ref(),
            ),
            graphics::create_swapchain(
                &xr_ctx.session,
                &gfx_ctx,
                config.view_resolution,
                format,
                foveation_profile.as_ref(),
            ),
        ];

        let renderer = StreamRenderer::new(
            gfx_ctx,
            config.view_resolution,
            [
                swapchains[0]
                    .enumerate_images()
                    .unwrap()
                    .iter()
                    .map(|i| *i as _)
                    .collect(),
                swapchains[1]
                    .enumerate_images()
                    .unwrap()
                    .iter()
                    .map(|i| *i as _)
                    .collect(),
            ],
            format,
            config.foveated_encoding_config.clone(),
            platform != Platform::Lynx
                && !((platform == Platform::Pico4 || platform == Platform::PicoNeo3)
                    && config.encoder_config.enable_hdr),
            !config.encoder_config.enable_hdr,
            config.encoder_config.encoding_gamma,
        );

        core_ctx.send_playspace(
            xr_ctx
                .session
                .reference_space_bounds_rect(xr::ReferenceSpaceType::STAGE)
                .unwrap()
                .map(|a| Vec2::new(a.width, a.height)),
        );

        core_ctx.send_active_interaction_profile(
            *HAND_LEFT_ID,
            interaction_ctx.hands_interaction[0].controllers_profile_id,
        );
        core_ctx.send_active_interaction_profile(
            *HAND_RIGHT_ID,
            interaction_ctx.hands_interaction[1].controllers_profile_id,
        );

        let input_thread_running = Arc::new(RelaxedAtomic::new(true));

        let reference_space = Arc::new(interaction::get_reference_space(
            &xr_ctx.session,
            xr::ReferenceSpaceType::STAGE,
        ));

        let input_thread = thread::spawn({
            let core_ctx = Arc::clone(&core_ctx);
            let xr_ctx = xr_ctx.clone();
            let interaction_ctx = Arc::clone(&interaction_ctx);
            let reference_space = Arc::clone(&reference_space);
            let refresh_rate = config.refresh_rate_hint;
            let running = Arc::clone(&input_thread_running);
            move || {
                stream_input_loop(
                    &core_ctx,
                    xr_ctx,
                    &interaction_ctx,
                    Arc::clone(&reference_space),
                    refresh_rate,
                    running,
                )
            }
        });

        StreamContext {
            core_context: core_ctx,
            xr_context: xr_ctx,
            interaction_context: interaction_ctx,
            reference_space,
            swapchains,
            view_resolution: config.view_resolution,
            refresh_rate: config.refresh_rate_hint,
            last_good_view_params: [ViewParams::default(); 2],
            input_thread: Some(input_thread),
            input_thread_running,
            renderer,
        }
    }

    pub fn update_reference_space(&mut self) {
        self.input_thread_running.set(false);

        self.reference_space = Arc::new(interaction::get_reference_space(
            &self.xr_context.session,
            xr::ReferenceSpaceType::STAGE,
        ));

        self.core_context.send_playspace(
            self.xr_context
                .session
                .reference_space_bounds_rect(xr::ReferenceSpaceType::STAGE)
                .unwrap()
                .map(|a| Vec2::new(a.width, a.height)),
        );

        if let Some(running) = self.input_thread.take() {
            running.join().ok();
        }

        self.input_thread_running.set(true);

        self.input_thread = Some(thread::spawn({
            let core_ctx = Arc::clone(&self.core_context);
            let xr_ctx = self.xr_context.clone();
            let interaction_ctx = Arc::clone(&self.interaction_context);
            let reference_space = Arc::clone(&self.reference_space);
            let refresh_rate = self.refresh_rate;
            let running = Arc::clone(&self.input_thread_running);
            move || {
                stream_input_loop(
                    &core_ctx,
                    xr_ctx,
                    &interaction_ctx,
                    Arc::clone(&reference_space),
                    refresh_rate,
                    running,
                )
            }
        }));
    }

    pub fn render(
        &mut self,
        decoded_frame: Option<DecodedFrame>,
        vsync_time: Duration,
    ) -> CompositionLayerBuilder {
        let timestamp;
        let view_params;
        let buffer_ptr;
        if let Some(frame) = decoded_frame {
            timestamp = frame.timestamp;
            view_params = frame.view_params;
            buffer_ptr = frame.buffer_ptr;

            self.last_good_view_params = frame.view_params;
        } else {
            timestamp = vsync_time;
            view_params = self.last_good_view_params;
            buffer_ptr = std::ptr::null_mut();
        }

        let left_swapchain_idx = self.swapchains[0].acquire_image().unwrap();
        let right_swapchain_idx = self.swapchains[1].acquire_image().unwrap();

        self.swapchains[0]
            .wait_image(xr::Duration::INFINITE)
            .unwrap();
        self.swapchains[1]
            .wait_image(xr::Duration::INFINITE)
            .unwrap();

        unsafe {
            self.renderer
                .render(buffer_ptr, [left_swapchain_idx, right_swapchain_idx])
        };

        self.swapchains[0].release_image().unwrap();
        self.swapchains[1].release_image().unwrap();

        if !buffer_ptr.is_null() {
            if let Some(now) = crate::xr_runtime_now(&self.xr_context.instance) {
                self.core_context
                    .report_submit(timestamp, vsync_time.saturating_sub(now));
            }
        }

        let rect = xr::Rect2Di {
            offset: xr::Offset2Di { x: 0, y: 0 },
            extent: xr::Extent2Di {
                width: self.view_resolution.x as _,
                height: self.view_resolution.y as _,
            },
        };

        CompositionLayerBuilder::new(
            &self.reference_space,
            [
                xr::CompositionLayerProjectionView::new()
                    .pose(to_xr_pose(view_params[0].pose))
                    .fov(to_xr_fov(view_params[0].fov))
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&self.swapchains[0])
                            .image_array_index(0)
                            .image_rect(rect),
                    ),
                xr::CompositionLayerProjectionView::new()
                    .pose(to_xr_pose(view_params[1].pose))
                    .fov(to_xr_fov(view_params[1].fov))
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&self.swapchains[1])
                            .image_array_index(0)
                            .image_rect(rect),
                    ),
            ],
        )
    }
}

impl Drop for StreamContext {
    fn drop(&mut self) {
        self.input_thread_running.set(false);
        self.input_thread.take().unwrap().join().ok();
    }
}

fn stream_input_loop(
    core_ctx: &ClientCoreContext,
    xr_ctx: XrContext,
    interaction_ctx: &InteractionContext,
    reference_space: Arc<xr::Space>,
    refresh_rate: f32,
    running: Arc<RelaxedAtomic>,
) {
    let mut last_controller_poses = [Pose::default(); 2];
    let mut last_palm_poses = [Pose::default(); 2];

    let mut deadline = Instant::now();
    let frame_interval = Duration::from_secs_f32(1.0 / refresh_rate);
    while running.value() {
        // Streaming related inputs are updated here. Make sure every input poll is done in this
        // thread
        if let Err(e) = xr_ctx
            .session
            .sync_actions(&[(&interaction_ctx.action_set).into()])
        {
            error!("{e}");
            return;
        }

        let Some(now) = crate::xr_runtime_now(&xr_ctx.instance) else {
            error!("Cannot poll tracking: invalid time");
            return;
        };

        let target_timestamp =
            now + Duration::min(core_ctx.get_head_prediction_offset(), MAX_PREDICTION);

        let Ok((view_flags, views)) = xr_ctx.session.locate_views(
            xr::ViewConfigurationType::PRIMARY_STEREO,
            crate::to_xr_time(target_timestamp),
            &reference_space,
        ) else {
            error!("Cannot locate views");
            continue;
        };

        if !view_flags.contains(xr::ViewStateFlags::POSITION_VALID)
            || !view_flags.contains(xr::ViewStateFlags::ORIENTATION_VALID)
        {
            continue;
        }

        let view_params = [
            ViewParams {
                pose: from_xr_pose(views[0].pose),
                fov: crate::from_xr_fov(views[0].fov),
            },
            ViewParams {
                pose: from_xr_pose(views[1].pose),
                fov: crate::from_xr_fov(views[1].fov),
            },
        ];

        let mut device_motions = Vec::with_capacity(3);

        let tracker_time = crate::to_xr_time(
            now + Duration::min(core_ctx.get_tracker_prediction_offset(), MAX_PREDICTION),
        );

        let (left_hand_motion, left_hand_skeleton) = crate::interaction::get_hand_data(
            &xr_ctx.session,
            &reference_space,
            tracker_time,
            &interaction_ctx.hands_interaction[0],
            &mut last_controller_poses[0],
            &mut last_palm_poses[0],
        );
        let (right_hand_motion, right_hand_skeleton) = crate::interaction::get_hand_data(
            &xr_ctx.session,
            &reference_space,
            tracker_time,
            &interaction_ctx.hands_interaction[1],
            &mut last_controller_poses[1],
            &mut last_palm_poses[1],
        );

        // Note: When multimodal input is enabled, we are sure that when free hands are used
        // (not holding controllers) the controller data is None.
        if interaction_ctx.uses_multimodal_hands || left_hand_skeleton.is_none() {
            if let Some(motion) = left_hand_motion {
                device_motions.push((*HAND_LEFT_ID, motion));
            }
        }
        if interaction_ctx.uses_multimodal_hands || right_hand_skeleton.is_none() {
            if let Some(motion) = right_hand_motion {
                device_motions.push((*HAND_RIGHT_ID, motion));
            }
        }

        let face_data = FaceData {
            eye_gazes: interaction::get_eye_gazes(
                &xr_ctx.session,
                &interaction_ctx.face_sources,
                &reference_space,
                crate::to_xr_time(now),
            ),
            fb_face_expression: interaction::get_fb_face_expression(
                &interaction_ctx.face_sources,
                crate::to_xr_time(now),
            ),
            htc_eye_expression: interaction::get_htc_eye_expression(&interaction_ctx.face_sources),
            htc_lip_expression: interaction::get_htc_lip_expression(&interaction_ctx.face_sources),
        };

        if let Some((tracker, joint_count)) = &interaction_ctx.body_sources.body_tracker_fb {
            device_motions.append(&mut interaction::get_fb_body_tracking_points(
                &reference_space,
                crate::to_xr_time(now),
                tracker,
                *joint_count,
            ));
        }

        core_ctx.send_tracking(
            target_timestamp,
            view_params,
            device_motions,
            [left_hand_skeleton, right_hand_skeleton],
            face_data,
        );

        let button_entries =
            interaction::update_buttons(&xr_ctx.session, &interaction_ctx.button_actions);
        if !button_entries.is_empty() {

            // If any of the button entries contains a true binary value
            // we need to send the updated button states to the server
            if button_entries.iter().any(|entry| matches!(entry.value, ButtonValue::Binary(true))) {
                let xr_session = &xr_ctx.session;
                let view = xr_session.locate_views(
                    ViewConfigurationType::PRIMARY_STEREO, tracker_time, &&interaction::get_reference_space(xr_session, xr::ReferenceSpaceType::LOCAL))
                    .unwrap().1[0];

                if let Err(e) = create_spatial_anchor(
                    &xr_session.instance(),
                    xr_session,
                    interaction::get_reference_space(xr_session, xr::ReferenceSpaceType::LOCAL).as_raw(),
                    view.pose.position,
                    view.pose.orientation,
                    tracker_time
                ) {
                    eprintln!("Failed to create spatial anchor: {:?}", e);
                } else {
                    println!("Spatial anchor created successfully with position and orientation: {:?}, {:?}", view.pose.position, view.pose.orientation);
                }

                if let Ok(poses) = query_spatial_anchors(
                    &xr_session.instance(),
                    xr_session,
                    interaction::get_reference_space(xr_session, xr::ReferenceSpaceType::LOCAL).as_raw(),
                    tracker_time
                ) {
                    for pose in poses {
                        println!("Spatial anchor found with position and orientation: {:?}, {:?}", pose.position, pose.orientation);
                    }
                } else {
                    eprintln!("Failed to query spatial anchor");
                }
            }   

            core_ctx.send_buttons(button_entries);
         
        }

        deadline += frame_interval / 3;
        thread::sleep(deadline.saturating_duration_since(Instant::now()));
    }
}

fn query_spatial_anchors(
    xr_instance: &openxr::Instance,
    session: &openxr::Session<openxr::OpenGlEs>,
    space: sys::Space,
    time: sys::Time
) -> Result<Vec<Pose>, openxr::sys::Result> {
    
    let anchor_query_info = sys::SpaceQueryInfoBaseHeaderFB {
        ty: sys::StructureType::SPACE_QUERY_INFO_FB,
        next: std::ptr::null(),
    };

    let mut space_query_results = sys::SpaceQueryResultsFB {
        ty: sys::StructureType::SPACE_QUERY_RESULTS_FB,
        next: std::ptr::null::<sys::SpaceQueryResultsFB>() as *mut _,
        result_capacity_input: 0,
        result_count_output: 0,
        results: std::ptr::null::<sys::SpaceQueryResultFB>() as *mut _,
    };

    let mut async_id : sys::AsyncRequestIdFB = Default::default();

    let result = unsafe {
        if let Some(query_spatial_anchor_fb) = xr_instance.exts().fb_spatial_entity_query {
            (query_spatial_anchor_fb.query_spaces)(session.as_raw(), &anchor_query_info, &mut async_id)
        } else {
            eprintln!("Failed to query spatial anchor: fb_spatial_entity_query extension not available");
            return Err(sys::Result::ERROR_EXTENSION_NOT_PRESENT);
        }
    };

    if result == sys::Result::SUCCESS {
        let result2 = unsafe {
            if let Some(query_spatial_anchor_fb) = xr_instance.exts().fb_spatial_entity_query {
                (query_spatial_anchor_fb.retrieve_space_query_results)(session.as_raw(), async_id, &mut space_query_results)
            } else {
                eprintln!("Failed to query spatial anchor: fb_spatial_entity_query extension not available");
                return Err(sys::Result::ERROR_EXTENSION_NOT_PRESENT);
            }
        };
        if result2 == sys::Result::SUCCESS {
            let mut poses = Vec::new();
            println!("Retrieved spatial anchor query results number: {:?}", space_query_results.result_count_output);
            for i in 0..space_query_results.result_count_output {
                let space_query_result_fb = unsafe { *space_query_results.results.offset(i as isize) };
                let spatial_anchor = space_query_result_fb.space;
                let mut spatial_anchor_location = sys::SpaceLocation{
                    ty: sys::StructureType::SPACE_LOCATION,
                    next: std::ptr::null::<sys::SpaceLocation>() as *mut _,
                    location_flags: sys::SpaceLocationFlags::POSITION_VALID | sys::SpaceLocationFlags::ORIENTATION_VALID,
                    pose: sys::Posef {
                        position: openxr::Vector3f { x: 0.0, y: 0.0, z: 0.0 },
                        orientation: openxr::Quaternionf::IDENTITY
                    }
                };

                println!("Locating space...");

                let result3 = unsafe {
                    (xr_instance.fp().locate_space)(
                        spatial_anchor,
                        space,
                        time,
                        &mut spatial_anchor_location,
                    )
                };

                if result3 == sys::Result::SUCCESS {
                    println!("Constructing poses");
                    poses.push(Pose {
                        position: Vec3::new(
                            spatial_anchor_location.pose.position.x,
                            spatial_anchor_location.pose.position.y,
                            spatial_anchor_location.pose.position.z,
                        ),
                        orientation: glam::Quat::from_xyzw(
                            spatial_anchor_location.pose.orientation.x,
                            spatial_anchor_location.pose.orientation.y,
                            spatial_anchor_location.pose.orientation.z,
                            spatial_anchor_location.pose.orientation.w,
                        ),
                    });
                } else {
                    eprintln!("Failed to locate space: {:?}", result3);
                }
                
            }
            println!("poses: {:?}", poses); 
            Ok(poses)
        } else {
            Err(result2)
        }

    } else {
        Err(result)
    }
}

fn create_spatial_anchor(
    xr_instance: &openxr::Instance,
    session: &openxr::Session<openxr::OpenGlEs>,
    space: sys::Space,
    position: openxr::Vector3f,
    orientation: openxr::Quaternionf,
    time: sys::Time
) -> Result<openxr::sys::Space, openxr::sys::Result> {

    fn handle_spatial_anchor_event(
        xr_instance: &openxr::Instance
    ) -> Option<sys::Space> {
        let mut event_data = xr::EventDataBuffer::new();

        while let Ok(result) = xr_instance.poll_event(&mut event_data) {
            
            if let Some(event) = result {
                if let xr::Event::SpatialAnchorCreateCompleteFB(event) = event {
                    println!("Spatial anchor saved successfully with space handle: {:?}", event.space());
                    return Some(event.space())
                }
            }
        }

        None
    }

    fn handle_spatial_anchor_component_event(
        xr_instance: &openxr::Instance
    ) -> Option<sys::Space> {
        let mut event_data = xr::EventDataBuffer::new();

        while let Ok(result) = xr_instance.poll_event(&mut event_data) {
            
            if let Some(event) = result {
                if let xr::Event::SpaceSetStatusCompleteFB(event) = event {
                    println!("Spatial anchor local component set succesfully");
                    return Some(event.space())
                }
            }
        }

        None
    }

    fn handle_spatial_anchor_save_event(
        xr_instance: &openxr::Instance
    ) -> Option<sys::Space> {
        let mut event_data = xr::EventDataBuffer::new();

        while let Ok(result) = xr_instance.poll_event(&mut event_data) {
            
            if let Some(event) = result {
                if let xr::Event::SpaceSaveCompleteFB(event) = event {
                    println!("Spatial anchor saved succesfully");
                    return Some(event.space())
                }
            }
        }

        None
    }

    // Prepare the spatial anchor create info
    let anchor_create_info = sys::SpatialAnchorCreateInfoFB {
        ty: sys::StructureType::SPATIAL_ANCHOR_CREATE_INFO_FB,
        next: std::ptr::null(),
        space,
        pose_in_space: sys::Posef {
            position,
            orientation,
        },
        time,
    };

    // Initialize a variable to receive the request ID
    let mut request_id: sys::AsyncRequestIdFB = Default::default();

    // Call the function to create the spatial anchor
    let result = unsafe {
        if let Some(create_spatial_anchor_fn) = xr_instance.exts().fb_spatial_entity {
            (create_spatial_anchor_fn.create_spatial_anchor)(session.as_raw(), &anchor_create_info, &mut request_id)
        } else {
            eprintln!("Failed to create spatial anchor: fb_spatial_entity extension not available");
            return Err(sys::Result::ERROR_EXTENSION_NOT_PRESENT);
        }
    };

    if result == sys::Result::SUCCESS {

        let spatial_anchor = loop {
            println!("Polling for spatial anchor event...");
            if let Some(anchor) = handle_spatial_anchor_event(xr_instance) {
                break anchor
            }
            println!("Waiting for spatial anchor event...");
        };

        // Print the spatial anchor handle
        println!("Spatial anchor handle: {:?}", spatial_anchor);

        let mut space_component_request_id: sys::AsyncRequestIdFB = Default::default();
        let mut space_component_status = sys::SpaceComponentStatusSetInfoFB {
            ty: sys::StructureType::SPACE_COMPONENT_STATUS_SET_INFO_FB,
            next: std::ptr::null(),
            component_type: sys::SpaceComponentTypeFB::STORABLE,
            enabled: sys::TRUE,
            timeout: xr::Duration::INFINITE

        };

        // Call the function to create the spatial anchor
        let result2 = unsafe {
            if let Some(create_spatial_anchor_fn) = xr_instance.exts().fb_spatial_entity {
                (create_spatial_anchor_fn.set_space_component_status)(spatial_anchor, &mut space_component_status, &mut space_component_request_id)
            } else {
                eprintln!("Failed to create spatial anchor: fb_spatial_entity extension not available");
                return Err(sys::Result::ERROR_EXTENSION_NOT_PRESENT);
            }
        };

        if result2 == sys::Result::SUCCESS {

            let spatial_anchor = loop {
                println!("Polling for spatial anchor component event...");
                if let Some(anchor) = handle_spatial_anchor_component_event(xr_instance) {
                    break anchor
                }
                println!("Waiting for spatial anchor component event...");
            };

            let mut save_space_info = sys::SpaceSaveInfoFB {
                ty: sys::StructureType::SPACE_SAVE_INFO_FB,
                next: std::ptr::null(),
                space: spatial_anchor,
                location: sys::SpaceStorageLocationFB::LOCAL,
                persistence_mode: sys::SpacePersistenceModeFB::INDEFINITE
    
            };
    
            let result3 = unsafe {
                if let Some(save_space_fn) = xr_instance.exts().fb_spatial_entity_storage {
                    println!("{:?}", save_space_info);
                    (save_space_fn.save_space)(session.as_raw(), &mut save_space_info, &mut request_id)
                } else {
                    eprintln!("Failed to save spatial anchor: fb_spatial_entity_storage extension not available");
                    return Err(sys::Result::ERROR_EXTENSION_NOT_PRESENT);
                }
    
            };

            if result3 == sys::Result::SUCCESS {

                let spatial_anchor = loop {
                    println!("Polling for spatial anchor save event...");
                    if let Some(anchor) = handle_spatial_anchor_save_event(xr_instance) {
                        break anchor
                    }
                    println!("Waiting for spatial anchor save event...");
                };

                println!("CREATE SPATIAL ANCHOR COMPLETE SUCCESS");
                Ok(spatial_anchor)
            } else {
                Err(result3)
            }

        } else {
            Err(result2)
        }
        
    } else {
        Err(result)
    }
}
