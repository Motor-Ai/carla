agent_feat_id = {
    "id" : 0, #tracking id
    "x": 1,  # m
    "y": 2,  # m
    "yaw": 3,  # rad
    "vx": 4,  # m/s
    "vy": 5,  # m/s
    "length": 6,  # m
    "width": 7,  # m
    "height": 8,  # m
    "class": 9,  # class id
    "is_dynamic": 10,  # boolean
    "road_id": 11,  # int
    "lane_id": 12,  # int
    "is_junction": 13,  # boolean TODO verify
    "s" : 14 # int
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