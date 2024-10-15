import argparse
import math
import threading
import queue

from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server

import open3d as o3d
import numpy as np


def custom_handler(address, *args):
    # Extract the arguments
    # Arguments contains the id, x, y, z, qx, qy, qz, qw 

    number_of_anchors = int(len(args) / 8)

    anchors = []

    for i in range(number_of_anchors):
        id = args[i * 8]
        x = args[i * 8 + 1]
        y = args[i * 8 + 2]
        z = args[i * 8 + 3]
        qx = args[i * 8 + 4]
        qy = args[i * 8 + 5]
        qz = args[i * 8 + 6]
        qw = args[i * 8 + 7]

        anchors.append([x, y, z])

    # Put the anchors in the queue
    data_queue.put(anchors)


def visualizer_thread():
    global vis, pcd

    # Initialize Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    
    vis.add_geometry(pcd)

    #Make point size bigger
    vis.get_render_option().point_size = 10
    first_time = True

    while True:
        # Get the anchors from the queue
        anchors = data_queue.get()

        # Update the point cloud
        points = np.array(anchors)

        # Convert from y up x right z back to y back x right z up
        points = points[:, [2, 0, 1]]

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(anchors), 3)))
        # Recenter the view
        if first_time:
            vis.reset_view_point(True)
            first_time = False

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        data_queue.task_done()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",
                        default="172.16.4.126", help="The ip to listen on")
    parser.add_argument("--port",
                        type=int, default=8002, help="The port to listen on")
    args = parser.parse_args()

    dispatcher = Dispatcher()
    dispatcher.map("/spatial_anchors", custom_handler)

    server = osc_server.ThreadingOSCUDPServer(
        (args.ip, args.port), dispatcher)

    print("Serving on {}".format(server.server_address))

    # Create a queue to communicate between threads
    data_queue = queue.Queue()

    # Start the visualizer thread
    vis_thread = threading.Thread(target=visualizer_thread)
    vis_thread.daemon = True
    vis_thread.start()

    print("Visualizer thread started")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()
