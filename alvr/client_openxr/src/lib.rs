mod c_api;
mod extra_extensions;
mod graphics;
mod interaction;
mod lobby;
mod stream;

use crate::stream::ParsedStreamConfig;
use alvr_client_core::{
    graphics::GraphicsContext, ClientCapabilities, ClientCoreContext, ClientCoreEvent, Platform,
};
use alvr_common::{
    error,
    glam::{self, Quat, UVec2, Vec3},
    info, Fov, Pose, HAND_LEFT_ID,
};
use extra_extensions::{
    META_BODY_TRACKING_FULL_BODY_EXTENSION_NAME, META_DETACHED_CONTROLLERS_EXTENSION_NAME,
    META_SIMULTANEOUS_HANDS_AND_CONTROLLERS_EXTENSION_NAME,
};
use lobby::Lobby;
use openxr::{self as xr, sys, ViewConfigurationType};
use std::{
    path::Path,
    rc::Rc,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
use stream::StreamContext;

use rosc::encoder;
use rosc::{OscMessage, OscPacket, OscType};
use std::net::{SocketAddrV4, UdpSocket};

const DECODER_MAX_TIMEOUT_MULTIPLIER: f32 = 0.8;

fn from_xr_vec3(v: xr::Vector3f) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

fn to_xr_vec3(v: Vec3) -> xr::Vector3f {
    xr::Vector3f {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}

fn from_xr_quat(q: xr::Quaternionf) -> Quat {
    Quat::from_xyzw(q.x, q.y, q.z, q.w)
}

fn to_xr_quat(q: Quat) -> xr::Quaternionf {
    xr::Quaternionf {
        x: q.x,
        y: q.y,
        z: q.z,
        w: q.w,
    }
}

fn from_xr_pose(p: xr::Posef) -> Pose {
    Pose {
        orientation: from_xr_quat(p.orientation),
        position: from_xr_vec3(p.position),
    }
}

fn to_xr_pose(p: Pose) -> xr::Posef {
    xr::Posef {
        orientation: to_xr_quat(p.orientation),
        position: to_xr_vec3(p.position),
    }
}

fn from_xr_fov(f: xr::Fovf) -> Fov {
    Fov {
        left: f.angle_left,
        right: f.angle_right,
        up: f.angle_up,
        down: f.angle_down,
    }
}

fn to_xr_fov(f: Fov) -> xr::Fovf {
    xr::Fovf {
        angle_left: f.left,
        angle_right: f.right,
        angle_up: f.up,
        angle_down: f.down,
    }
}

fn to_xr_time(timestamp: Duration) -> xr::Time {
    xr::Time::from_nanos(timestamp.as_nanos() as _)
}

#[derive(Clone)]
pub struct XrContext {
    instance: xr::Instance,
    system: xr::SystemId,
    session: xr::Session<xr::OpenGlEs>,
}

fn default_view() -> xr::View {
    xr::View {
        pose: xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: xr::Vector3f::default(),
        },
        fov: xr::Fovf {
            angle_left: -1.0,
            angle_right: 1.0,
            angle_up: 1.0,
            angle_down: -1.0,
        },
    }
}

fn query_spatial_anchors(
    xr_instance: &openxr::Instance,
    session: &openxr::Session<openxr::OpenGlEs>
) -> Result<Vec<(sys::Space, sys::UuidEXT)>, openxr::sys::Result> {

    fn handle_spatial_anchor_query_event(
        xr_instance: &openxr::Instance
    ) -> Option<i32> {
        let mut event_data = xr::EventDataBuffer::new();

        while let Ok(result) = xr_instance.poll_event(&mut event_data) {
            
            if let Some(event) = result {
                if let xr::Event::SpaceQueryResultsAvailableFB(event) = event {
                    println!("Spatial anchors query completed successfully");
                    return Some(1)
                }
                else if let xr::Event::SpaceQueryCompleteFB(event) = event {
                    println!("Spatial anchors query completed successfully");
                    return Some(1)
                    
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
    
    let anchor_query_info = sys::SpaceQueryInfoFB {
        ty: sys::StructureType::SPACE_QUERY_INFO_FB,
        next: std::ptr::null(),
        query_action: sys::SpaceQueryActionFB::LOAD,
        max_result_count: 1000,
        timeout: openxr::Duration::NONE,
        filter: std::ptr::null(),
        exclude_filter: std::ptr::null()
    };

    let mut async_id : sys::AsyncRequestIdFB = Default::default();

    let result = unsafe {
        if let Some(query_spatial_anchor_fb) = xr_instance.exts().fb_spatial_entity_query {
            (query_spatial_anchor_fb.query_spaces)(session.as_raw(), &anchor_query_info as *const _ as *const sys::SpaceQueryInfoBaseHeaderFB, &mut async_id)
        } else {
            eprintln!("Failed to query spatial anchor: fb_spatial_entity_query extension not available");
            return Err(sys::Result::ERROR_EXTENSION_NOT_PRESENT);
        }
    };

    if result == sys::Result::SUCCESS {

        loop {
            println!("Polling for spatial anchor query event...");
            if let Some(anchor) = handle_spatial_anchor_query_event(xr_instance) {
                break;
            }
            println!("Waiting for spatial anchor query event...");
        };

        let mut space_query_results = sys::SpaceQueryResultsFB {
            ty: sys::StructureType::SPACE_QUERY_RESULTS_FB,
            next: std::ptr::null::<sys::SpaceQueryResultsFB>() as *mut _,
            result_capacity_input: 0,
            result_count_output: 0,
            results: std::ptr::null::<sys::SpaceQueryResultFB>() as *mut _,
        };

        let result2 = unsafe {
            if let Some(query_spatial_anchor_fb) = xr_instance.exts().fb_spatial_entity_query {

                (query_spatial_anchor_fb.retrieve_space_query_results)(session.as_raw(), async_id, &mut space_query_results);
                
                let num_results = space_query_results.result_count_output;

                let mut results_buffer: Vec<sys::SpaceQueryResultFB> = vec![
                    sys::SpaceQueryResultFB {
                        space: sys::Space::NULL,
                        uuid: sys::UuidEXT {
                            data: [0; sys::UUID_SIZE_EXT]
                        },
                    };
                    num_results as usize
                ];                

                space_query_results = sys::SpaceQueryResultsFB {
                    ty: sys::StructureType::SPACE_QUERY_RESULTS_FB,
                    next: std::ptr::null::<sys::SpaceQueryResultsFB>() as *mut _,
                    result_capacity_input: num_results,
                    result_count_output: num_results,
                    // Create a buffer to store the results as big as the number of results
                    results: results_buffer.as_mut_ptr(),
                };

                (query_spatial_anchor_fb.retrieve_space_query_results)(session.as_raw(), async_id, &mut space_query_results)

            } else {
                eprintln!("Failed to query spatial anchor: fb_spatial_entity_query extension not available");
                return Err(sys::Result::ERROR_EXTENSION_NOT_PRESENT);
            }
        };
        if result2 == sys::Result::SUCCESS {
            let mut spatial_anchors = Vec::new();
            println!("Retrieved spatial anchor query results number: {:?}", space_query_results.result_count_output);
            println!("Retrieved spatial anchor query results capacity: {:?}", space_query_results.result_capacity_input);
            let create_spatial_anchor_fn = xr_instance.exts().fb_spatial_entity;

            for i in 0..space_query_results.result_count_output {
                let space_query_result_fb = unsafe { space_query_results.results.add(i as usize).read() };
                let spatial_anchor = space_query_result_fb.space;
                let spatial_anchor_id = space_query_result_fb.uuid;

                let mut space_component_request_id: sys::AsyncRequestIdFB = Default::default();
                let mut space_component_status = sys::SpaceComponentStatusSetInfoFB {
                    ty: sys::StructureType::SPACE_COMPONENT_STATUS_SET_INFO_FB,
                    next: std::ptr::null(),
                    component_type: sys::SpaceComponentTypeFB::LOCATABLE,
                    enabled: sys::TRUE,
                    timeout: xr::Duration::INFINITE

                };

                // Call the function to create the spatial anchor
                let result3 = unsafe {
                    if let Some(create_spatial_anchor_fn) = create_spatial_anchor_fn {
                        (create_spatial_anchor_fn.set_space_component_status)(spatial_anchor, &mut space_component_status, &mut space_component_request_id)
                    } else {
                        eprintln!("Failed to create spatial anchor: fb_spatial_entity extension not available");
                        return Err(sys::Result::ERROR_EXTENSION_NOT_PRESENT);
                    }
                };

                if result3 == sys::Result::SUCCESS {

                    let spatial_anchor = loop {
                        println!("Polling for spatial anchor component event...");
                        if let Some(anchor) = handle_spatial_anchor_component_event(xr_instance) {
                            break anchor
                        }
                        println!("Waiting for spatial anchor component event...");
                    };

                    spatial_anchors.push((spatial_anchor, spatial_anchor_id));
             
                } else {
                    eprintln!("Failed to set spatial anchor component status: {:?}", result3);
                }
            }
            Ok(spatial_anchors)
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
) -> Result<(openxr::sys::Space, sys::UuidEXT), openxr::sys::Result> {

    fn handle_spatial_anchor_event(
        xr_instance: &openxr::Instance
    ) -> Option<(openxr::sys::Space, sys::UuidEXT)> {
        let mut event_data = xr::EventDataBuffer::new();

        while let Ok(result) = xr_instance.poll_event(&mut event_data) {
            
            if let Some(event) = result {
                if let xr::Event::SpatialAnchorCreateCompleteFB(event) = event {
                    println!("Spatial anchor saved successfully with space handle: {:?}", event.space());
                    return Some((event.space(), event.uuid()))
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

        let (spatial_anchor, spatial_anchor_id) = loop {
            println!("Polling for spatial anchor event...");
            if let Some((anchor, anchor_id)) = handle_spatial_anchor_event(xr_instance) {
                break (anchor, anchor_id)
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
                Ok((spatial_anchor, spatial_anchor_id))
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

fn locate_spatial_anchors(
    xr_instance: &openxr::Instance,
    spatial_anchor: sys::Space,
    space: sys::Space,
    time: sys::Time

) -> Result<Pose, openxr::sys::Result> {

    let mut spatial_anchor_location = sys::SpaceLocation{
        ty: sys::StructureType::SPACE_LOCATION,
        next: std::ptr::null::<sys::SpaceLocation>() as *mut _,
        location_flags: sys::SpaceLocationFlags::POSITION_VALID | sys::SpaceLocationFlags::ORIENTATION_VALID,
        pose: sys::Posef {
            position: openxr::Vector3f { x: 0.0, y: 0.0, z: 0.0 },
            orientation: openxr::Quaternionf::IDENTITY
        }
    };

    let result = unsafe {
        (xr_instance.fp().locate_space)(
            spatial_anchor,
            space,
            time,
            &mut spatial_anchor_location,
        )
    };

    if result == sys::Result::SUCCESS {
        Ok(Pose {
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
        })

    } else {
        eprintln!("Failed to locate space: {:?}", result);
        Err(result)
    } 
}

pub fn entry_point() {
    alvr_client_core::init_logging();

    let platform = alvr_client_core::platform();

    let loader_suffix = match platform {
        Platform::Quest1 => "_quest1",
        Platform::PicoNeo3 | Platform::Pico4 => "_pico",
        Platform::Yvr => "_yvr",
        Platform::Lynx => "_lynx",
        _ => "",
    };
    let xr_entry = unsafe {
        xr::Entry::load_from(Path::new(&format!("libopenxr_loader{loader_suffix}.so"))).unwrap()
    };

    #[cfg(target_os = "android")]
    xr_entry.initialize_android_loader().unwrap();

    let available_extensions = xr_entry.enumerate_extensions().unwrap();
    alvr_common::info!("OpenXR available extensions: {available_extensions:#?}");

    // todo: switch to vulkan
    assert!(available_extensions.khr_opengl_es_enable);

    let mut exts = xr::ExtensionSet::default();
    exts.bd_controller_interaction = available_extensions.bd_controller_interaction;
    exts.ext_eye_gaze_interaction = available_extensions.ext_eye_gaze_interaction;
    exts.ext_hand_tracking = available_extensions.ext_hand_tracking;
    exts.ext_local_floor = available_extensions.ext_local_floor;
    exts.fb_color_space = available_extensions.fb_color_space;
    exts.fb_display_refresh_rate = available_extensions.fb_display_refresh_rate;
    exts.fb_eye_tracking_social = available_extensions.fb_eye_tracking_social;
    exts.fb_face_tracking2 = available_extensions.fb_face_tracking2;
    exts.fb_body_tracking = available_extensions.fb_body_tracking;
    exts.fb_foveation = available_extensions.fb_foveation;
    exts.fb_foveation_configuration = available_extensions.fb_foveation_configuration;
    exts.fb_swapchain_update_state = available_extensions.fb_swapchain_update_state;
    exts.htc_facial_tracking = available_extensions.htc_facial_tracking;
    exts.htc_vive_focus3_controller_interaction =
        available_extensions.htc_vive_focus3_controller_interaction;
    #[cfg(target_os = "android")]
    {
        exts.khr_android_create_instance = true;
    }
    exts.khr_convert_timespec_time = true;
    exts.khr_opengl_es_enable = true;
    exts.fb_spatial_entity = available_extensions.fb_spatial_entity;
    exts.fb_spatial_entity_query = available_extensions.fb_spatial_entity_query;
    exts.fb_spatial_entity_storage = available_extensions.fb_spatial_entity_storage;
    exts.other = available_extensions
        .other
        .into_iter()
        .filter(|ext| {
            [
                META_BODY_TRACKING_FULL_BODY_EXTENSION_NAME,
                META_SIMULTANEOUS_HANDS_AND_CONTROLLERS_EXTENSION_NAME,
                META_DETACHED_CONTROLLERS_EXTENSION_NAME,
            ]
            .contains(&ext.as_str())
        })
        .collect();

    let available_layers = xr_entry.enumerate_layers().unwrap();
    alvr_common::info!("OpenXR available layers: {available_layers:#?}");

    let xr_instance = xr_entry
        .create_instance(
            &xr::ApplicationInfo {
                application_name: "ALVR Client",
                application_version: 0,
                engine_name: "ALVR",
                engine_version: 0,
            },
            &exts,
            &[],
        )
        .unwrap();

    let graphics_context = Rc::new(GraphicsContext::new_gl());

    let mut last_lobby_message = String::new();
    let mut parsed_stream_config = None::<ParsedStreamConfig>;

    'session_loop: loop {
        let xr_system = xr_instance
            .system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
            .unwrap();

        // mandatory call
        let _ = xr_instance
            .graphics_requirements::<xr::OpenGlEs>(xr_system)
            .unwrap();

        let (xr_session, mut xr_frame_waiter, mut xr_frame_stream) = unsafe {
            xr_instance
                .create_session(xr_system, &graphics::session_create_info(&graphics_context))
                .unwrap()
        };

        let xr_context = XrContext {
            instance: xr_instance.clone(),
            system: xr_system,
            session: xr_session.clone(),
        };

        let views_config = xr_instance
            .enumerate_view_configuration_views(
                xr_system,
                xr::ViewConfigurationType::PRIMARY_STEREO,
            )
            .unwrap();
        assert_eq!(views_config.len(), 2);

        let default_view_resolution = UVec2::new(
            views_config[0].recommended_image_rect_width,
            views_config[0].recommended_image_rect_height,
        );

        let refresh_rates = if exts.fb_display_refresh_rate {
            xr_session.enumerate_display_refresh_rates().unwrap()
        } else {
            vec![90.0]
        };

        if exts.fb_color_space {
            xr_session.set_color_space(xr::ColorSpaceFB::P3).unwrap();
        }

        let capabilities = ClientCapabilities {
            default_view_resolution,
            external_decoder: false,
            refresh_rates,
            foveated_encoding: platform != Platform::Unknown,
            encoder_high_profile: platform != Platform::Unknown,
            encoder_10_bits: platform != Platform::Unknown,
            encoder_av1: platform == Platform::Quest3,
        };
        let core_context = Arc::new(ClientCoreContext::new(capabilities));

        let interaction_context = Arc::new(interaction::initialize_interaction(
            &xr_context,
            platform,
            parsed_stream_config
                .as_ref()
                .map(|c| c.prefers_multimodal_input)
                .unwrap_or(false),
            parsed_stream_config
                .as_ref()
                .and_then(|c| c.face_sources_config.clone()),
            parsed_stream_config
                .as_ref()
                .and_then(|c| c.body_sources_config.clone()),
        ));

        let mut lobby = Lobby::new(
            &xr_context,
            Rc::clone(&graphics_context),
            Arc::clone(&interaction_context),
            default_view_resolution,
            &last_lobby_message,
        );
        let mut session_running = false;
        let mut stream_context = None::<StreamContext>;

        let mut event_storage = xr::EventDataBuffer::new();
        
        //CUSTOM VARIABLES
        let mut spatial_anchors: Vec<(xr::sys::Space, sys::UuidEXT)> = Vec::new();
        let mut last_anchor_place_time = Instant::now();
        let host_address = "172.16.4.120:8002";
        let socket = UdpSocket::bind(host_address).expect("Couldn't bind to address");
        let target_address = "172.16.4.126:8002".parse::<SocketAddrV4>().expect("Couldn't parse target address");

        'render_loop: loop {
            while let Some(event) = xr_instance.poll_event(&mut event_storage).unwrap() {
                match event {
                    xr::Event::EventsLost(event) => {
                        error!("OpenXR: lost {} events!", event.lost_event_count());
                    }
                    xr::Event::InstanceLossPending(_) => break 'session_loop,
                    xr::Event::SessionStateChanged(event) => match event.state() {
                        xr::SessionState::READY => {
                            xr_session
                                .begin(xr::ViewConfigurationType::PRIMARY_STEREO)
                                .unwrap();

                            core_context.resume();

                            session_running = true;
                        }
                        xr::SessionState::STOPPING => {
                            session_running = false;

                            core_context.pause();

                            xr_session.end().unwrap();
                        }
                        xr::SessionState::EXITING => break 'render_loop,
                        xr::SessionState::LOSS_PENDING => break 'render_loop,
                        xr::SessionState::FOCUSED => {

                            let result = query_spatial_anchors(&xr_instance, &xr_session);

                            match result {
                                Ok(anchors) => {
                                    spatial_anchors = anchors;
                                    println!("Spatial anchors queried successfully");
                                }
                                Err(e) => {
                                    eprintln!("Failed to query spatial anchors: {:?}", e);
                                }
                            }  
                        }
                        _ => (),
                    },
                    xr::Event::ReferenceSpaceChangePending(event) => {
                        info!(
                            "ReferenceSpaceChangePending type: {:?}",
                            event.reference_space_type()
                        );

                        lobby.update_reference_space();

                        if let Some(context) = &mut stream_context {
                            context.update_reference_space();
                        }
                    }
                    xr::Event::PerfSettingsEXT(event) => {
                        info!(
                            "Perf: from {:?} to {:?}, domain: {:?}/{:?}",
                            event.from_level(),
                            event.to_level(),
                            event.domain(),
                            event.sub_domain(),
                        );
                    }
                    xr::Event::InteractionProfileChanged(_) => {
                        // todo
                    }
                    xr::Event::PassthroughStateChangedFB(_) => {
                        // todo
                    }
                    _ => (),
                }
            }

            if !session_running {
                thread::sleep(Duration::from_millis(100));
                continue;
            }

            while let Some(event) = core_context.poll_event() {
                match event {
                    ClientCoreEvent::UpdateHudMessage(message) => {
                        last_lobby_message.clone_from(&message);
                        lobby.update_hud_message(&message);
                    }
                    ClientCoreEvent::StreamingStarted(config) => {
                        let new_config = ParsedStreamConfig::new(&config);

                        // combined_eye_gaze is a setting that needs to be enabled at session
                        // creation. Since HTC headsets don't support session reinitialization, skip
                        // all elements that need it, that is face and eye tracking.
                        if parsed_stream_config.as_ref() != Some(&new_config)
                            && !matches!(
                                platform,
                                Platform::Focus3 | Platform::XRElite | Platform::ViveUnknown
                            )
                        {
                            parsed_stream_config = Some(new_config);

                            xr_session.request_exit().ok();
                        } else {
                            stream_context = Some(StreamContext::new(
                                Arc::clone(&core_context),
                                xr_context.clone(),
                                Rc::clone(&graphics_context),
                                Arc::clone(&interaction_context),
                                platform,
                                &new_config,
                            ));

                            parsed_stream_config = Some(new_config);
                        }
                    }
                    ClientCoreEvent::StreamingStopped => {
                        stream_context = None;
                    }
                    ClientCoreEvent::Haptics {
                        device_id,
                        duration,
                        frequency,
                        amplitude,
                    } => {
                        let action = if device_id == *HAND_LEFT_ID {
                            &interaction_context.hands_interaction[0].vibration_action
                        } else {
                            &interaction_context.hands_interaction[1].vibration_action
                        };

                        action
                            .apply_feedback(
                                &xr_session,
                                xr::Path::NULL,
                                &xr::HapticVibration::new()
                                    .amplitude(amplitude.clamp(0.0, 1.0))
                                    .frequency(frequency.max(0.0))
                                    .duration(xr::Duration::from_nanos(duration.as_nanos() as _)),
                            )
                            .unwrap();
                    }
                    ClientCoreEvent::DecoderConfig { .. } | ClientCoreEvent::FrameReady { .. } => {
                        panic!()
                    }
                }
            }

            let frame_state = match xr_frame_waiter.wait() {
                Ok(state) => state,
                Err(e) => {
                    error!("{e}");
                    panic!();
                }
            };
            let frame_interval =
                Duration::from_nanos(frame_state.predicted_display_period.as_nanos() as _);
            let vsync_time =
                Duration::from_nanos(frame_state.predicted_display_time.as_nanos() as _);

            xr_frame_stream.begin().unwrap();

            if !frame_state.should_render {
                xr_frame_stream
                    .end(
                        frame_state.predicted_display_time,
                        xr::EnvironmentBlendMode::OPAQUE,
                        &[],
                    )
                    .unwrap();

                continue;
            }

            // todo: allow rendering lobby and stream layers at the same time and add cross fade
            let (layer, display_time) = if let Some(context) = &mut stream_context {
                let frame_poll_deadline = Instant::now()
                    + Duration::from_secs_f32(
                        frame_interval.as_secs_f32() * DECODER_MAX_TIMEOUT_MULTIPLIER,
                    );
                let mut frame_result = None;
                while frame_result.is_none() && Instant::now() < frame_poll_deadline {
                    frame_result = core_context.get_frame();
                    thread::yield_now();
                }

                let timestamp = frame_result
                    .as_ref()
                    .map(|r| r.timestamp)
                    .unwrap_or(vsync_time);

                let layer = context.render(frame_result, vsync_time);

                (layer, timestamp)
            } else {
                let layer = lobby.render(frame_state.predicted_display_time);

                (layer, vsync_time)
            };

            graphics_context.make_current();
            let res = xr_frame_stream.end(
                to_xr_time(display_time),
                xr::EnvironmentBlendMode::OPAQUE,
                &[&layer.build()],
            );

            if let Err(e) = res {
                let time = to_xr_time(display_time);
                error!("End frame failed! {e}, timestamp: {display_time:?}, time: {time:?}");

                xr_frame_stream
                    .end(
                        frame_state.predicted_display_time,
                        xr::EnvironmentBlendMode::OPAQUE,
                        &[],
                    )
                    .unwrap();
            }

            // Place an anchor every 10 seconds
            if Instant::now().duration_since(last_anchor_place_time) > Duration::from_secs(2) {
                
                let view = xr_session.locate_views(
                    ViewConfigurationType::PRIMARY_STEREO, to_xr_time(display_time), &&interaction::get_reference_space(&xr_session, xr::ReferenceSpaceType::LOCAL))
                    .unwrap().1[0];

                let anchor_position = xr::Vector3f {
                    x: view.pose.position.x,
                    y: view.pose.position.y,
                    z: view.pose.position.z,
                };

                let anchor_orientation = xr::Quaternionf {
                    x: view.pose.orientation.x,
                    y: view.pose.orientation.y,
                    z: view.pose.orientation.z,
                    w: view.pose.orientation.w,
                };

                let result = create_spatial_anchor(&xr_instance, &xr_session, interaction::get_reference_space(&xr_session, xr::ReferenceSpaceType::LOCAL).as_raw(), anchor_position, anchor_orientation, to_xr_time(display_time));
                match result {
                    Ok(anchor) => {
                        spatial_anchors.push(anchor);
                        println!("Spatial anchor created successfully");
                    }
                    Err(e) => {
                        eprintln!("Failed to create spatial anchor: {:?}", e);
                    }
                }  

                last_anchor_place_time = Instant::now();

            }

            // Locate the spatial anchors every 1 second

            let poses = spatial_anchors.iter().map(|(anchor_space, _)| {
                locate_spatial_anchors(&xr_instance, *anchor_space, interaction::get_reference_space(&xr_session, xr::ReferenceSpaceType::LOCAL).as_raw(), to_xr_time(display_time))
            });

            let mut args = Vec::new();
            for (i, pose_result) in poses.enumerate() {
            match pose_result {
                Ok(pose) => {
                args.push(OscType::Int(spatial_anchors[i].1.data[0] as i32));
                args.push(OscType::Float(pose.position.x));
                args.push(OscType::Float(pose.position.y));
                args.push(OscType::Float(pose.position.z));
                args.push(OscType::Float(pose.orientation.x));
                args.push(OscType::Float(pose.orientation.y));
                args.push(OscType::Float(pose.orientation.z));
                args.push(OscType::Float(pose.orientation.w));
                }
                Err(e) => {
                eprintln!("Failed to locate spatial anchor {}: {:?}", i, e);
                }
            }
            }

            if !args.is_empty() {
            let msg = OscMessage {
                addr: "/spatial_anchors".to_string(),
                args,
            };
            let packet = OscPacket::Message(msg);
            let encoded_msg = encoder::encode(&packet).unwrap();
            socket.send_to(&encoded_msg, target_address).expect("Couldn't send data");
            }
        }

    }

    // grapics_context is dropped here
}

#[allow(unused)]
fn xr_runtime_now(xr_instance: &xr::Instance) -> Option<Duration> {
    let time_nanos = xr_instance.now().ok()?.as_nanos();

    (time_nanos > 0).then(|| Duration::from_nanos(time_nanos as _))
}

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: android_activity::AndroidApp) {
    use android_activity::{InputStatus, MainEvent, PollEvent};

    let rendering_thread = thread::spawn(|| {
        // workaround for the Pico runtime
        let context = ndk_context::android_context();
        let vm = unsafe { jni::JavaVM::from_raw(context.vm().cast()) }.unwrap();
        let _env = vm.attach_current_thread().unwrap();

        entry_point();
    });

    let mut should_quit = false;
    while !should_quit {
        app.poll_events(Some(Duration::from_millis(100)), |event| match event {
            PollEvent::Main(MainEvent::Destroy) => {
                should_quit = true;
            }
            PollEvent::Main(MainEvent::InputAvailable) => {
                if let Ok(mut iter) = app.input_events_iter() {
                    while iter.next(|_| InputStatus::Unhandled) {}
                }
            }
            _ => (),
        });
    }

    // Note: the quit event is sent from OpenXR too, this will return rather quicly.
    rendering_thread.join().unwrap();
}
