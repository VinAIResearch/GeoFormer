import torch


def unique_with_inds(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


@torch.no_grad()
def find_knn(gpu_index, locs, neighbor=32):
    n_points = locs.shape[0]
    # Search with torch GPU using pre-allocated arrays
    new_d_torch_gpu = torch.zeros(n_points, neighbor, device=locs.device, dtype=torch.float32)
    new_i_torch_gpu = torch.zeros(n_points, neighbor, device=locs.device, dtype=torch.int64)

    gpu_index.add(locs)

    gpu_index.search(locs, neighbor, new_d_torch_gpu, new_i_torch_gpu)
    gpu_index.reset()
    new_d_torch_gpu = torch.sqrt(new_d_torch_gpu)

    return new_d_torch_gpu, new_i_torch_gpu


@torch.no_grad()
def cal_geodesic_single(
    gpu_index, pre_enc_inds, locs_float_, batch_offset_, max_step=32, neighbor=32, radius=0.1, n_queries=128
):

    batch_size = pre_enc_inds.shape[0]
    geo_dists = []
    for b in range(batch_size):
        start = batch_offset_[b]
        end = batch_offset_[b + 1]

        query_inds = pre_enc_inds[b][:n_queries]
        locs_float_b = locs_float_[start:end]

        n_points = end - start

        new_d_torch_gpu, new_i_torch_gpu = find_knn(gpu_index, locs_float_b, neighbor=neighbor)

        geo_dist = torch.zeros((n_queries, n_points), dtype=torch.float32, device=locs_float_.device) - 1
        visited = torch.zeros((n_queries, n_points), dtype=torch.bool, device=locs_float_.device)

        for q in range(n_queries):
            D_geo, I_geo = new_d_torch_gpu[query_inds[q]], new_i_torch_gpu[query_inds[q]]

            indices, distances = I_geo[1:].reshape(-1), D_geo[1:].reshape(-1)

            cond = ((distances <= radius) & (indices >= 0)).bool()

            distances = distances[cond]
            indices = indices[cond]

            for it in range(max_step):

                indices_unique, corres_inds = unique_with_inds(indices)
                distances_uniques = distances[corres_inds]

                inds = torch.nonzero((visited[q, indices_unique] is False)).view(-1)

                if len(inds) < neighbor // 2:
                    break
                indices_unique = indices_unique[inds]
                distances_uniques = distances_uniques[inds]

                geo_dist[q, indices_unique] = distances_uniques
                visited[q, indices_unique] = True

                D_geo, I_geo = new_d_torch_gpu[indices_unique][:, 1:], new_i_torch_gpu[indices_unique][:, 1:]

                D_geo_cumsum = D_geo + distances_uniques.unsqueeze(-1)

                indices, distances_local, distances_global = (
                    I_geo.reshape(-1),
                    D_geo.reshape(-1),
                    D_geo_cumsum.reshape(-1),
                )
                cond = (distances_local <= radius) & (indices >= 0)
                distances = distances_global[cond]
                indices = indices[cond]
        geo_dists.append(geo_dist)
        del new_d_torch_gpu, new_i_torch_gpu
    return geo_dists


# NOTE fastest way to cal geodesic distance
@torch.no_grad()
def cal_geodesic_vectorize(
    gpu_index, pre_enc_inds, locs_float_, batch_offset_, max_step=128, neighbor=64, radius=0.05, n_queries=128
):

    batch_size = pre_enc_inds.shape[0]
    geo_dists = []
    for b in range(batch_size):
        start = batch_offset_[b]
        end = batch_offset_[b + 1]

        query_inds = pre_enc_inds[b][:n_queries].long()
        locs_float_b = locs_float_[start:end]

        n_points = end - start

        distances_arr, indices_arr = find_knn(gpu_index, locs_float_b, neighbor=neighbor)

        # NOTE nearest neigbor is themself -> remove first element
        distances_arr = distances_arr[:, 1:]
        indices_arr = indices_arr[:, 1:]

        geo_dist = torch.zeros((n_queries, n_points), dtype=torch.float32, device=locs_float_.device) - 1
        visited = torch.zeros((n_queries, n_points), dtype=torch.int32, device=locs_float_.device)

        arange_tensor = torch.arange(0, n_queries, dtype=torch.long, device=locs_float_.device)

        geo_dist[arange_tensor, query_inds] = 0.0
        visited[arange_tensor, query_inds] = 1

        distances, indices = distances_arr[query_inds], indices_arr[query_inds]  # N_queries x n_neighbors

        cond = (distances <= radius) & (indices >= 0)  # N_queries x n_neighbors

        queries_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp
        points_inds = indices[queries_inds, neighbors_inds]  # n_temp
        points_distances = distances[queries_inds, neighbors_inds]  # n_temp

        for step in range(max_step):
            # NOTE find unique indices for each query
            stack_pointquery_inds = torch.stack([points_inds, queries_inds], dim=0)
            _, unique_inds = unique_with_inds(stack_pointquery_inds)

            points_inds = points_inds[unique_inds]
            queries_inds = queries_inds[unique_inds]
            points_distances = points_distances[unique_inds]

            # NOTE update geodesic and visited look-up table
            geo_dist[queries_inds, points_inds] = points_distances
            visited[queries_inds, points_inds] = 1

            # NOTE get new neighbors
            distances_new, indices_new = distances_arr[points_inds], indices_arr[points_inds]  # n_temp x n_neighbors
            distances_new_cumsum = distances_new + points_distances[:, None]  # n_temp x n_neighbors

            # NOTE trick to repeat queries indices for new neighbor
            queries_inds = queries_inds[:, None].repeat(1, neighbor - 1)  # n_temp x n_neighbors

            # NOTE condition: no visited and radius and indices
            visited_cond = visited[queries_inds.flatten(), indices_new.flatten()].reshape(*distances_new.shape)
            cond = (distances_new <= radius) & (indices_new >= 0) & (visited_cond == 0)  # n_temp x n_neighbors

            # NOTE filter
            temp_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True)  # n_temp2

            if len(temp_inds) == 0:  # no new points:
                break

            points_inds = indices_new[temp_inds, neighbors_inds]  # n_temp2
            points_distances = distances_new_cumsum[temp_inds, neighbors_inds]  # n_temp2
            queries_inds = queries_inds[temp_inds, neighbors_inds]  # n_temp2

        geo_dists.append(geo_dist)
    return geo_dists
