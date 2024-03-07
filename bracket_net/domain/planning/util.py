def get_p_opt(vanilla_astar, map_designs, start_maps, goal_maps, paths):
    if map_designs.shape[1] == 1:
        va_outputs = vanilla_astar(map_designs, start_maps, goal_maps)
        pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
        pathlen_model = paths.sum((1, 2, 3)).detach().cpu().numpy()
        p_opt = (pathlen_astar == pathlen_model).mean()
        return p_opt