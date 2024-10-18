agent_feat_id = {
    # "id" : 0, #tracking id
    "x": 0,  # m
    "y": 1,  # m
    "yaw": 2,  # rad
    "vx": 3,  # m/s
    "vy": 4,  # m/s
    "length": 5,  # m
    "width": 6,  # m
    "height": 7,  # m
    "class": 8,  # class id
    "is_dynamic": 9,  # boolean
    "road_id": 10,  # int
    "lane_id": 11,  # int
    "is_junction": 12,  # boolean TODO verify
    "s" : 13 # int
}

rss_feat_id = {
    "rss_obj_id": 0, # int
    "rss_status": 1, # TODO: unit
    "rss_long_current_dist": 2, # TODO: unit
    "rss_long_safe_dist": 3, # TODO: unit
    "rss_lat_current_right_dist": 4, # TODO: unit
    "rss_lat_safe_right_dist": 5, # TODO: unit
    "rss_lat_current_left_dist": 6, # TODO: unit
    "rss_lat_safe_left_dist": 7, # TODO: unit
}