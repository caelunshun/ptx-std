// TODO thread_id, block_id, cluster_id, block_dim,
// block_id_in_cluster, block_rank_in_cluster (1D), warp_id (1D), lane_id (1D),
// dynamic_smem_size, dynamic_smem_start() -> *mut u8, ...

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}
