use sysinfo::{System, SystemExt, CpuExt, ProcessExt, DiskExt}; // Added ProcessExt, DiskExt for more detailed info if needed later

// sysinfo returns memory values in KiB. Use this factor to convert KiB -> GB
const KIB_TO_GB: f32 = 1024.0 * 1024.0; // 1024 KiB * 1024 = 1 GiB

#[derive(Debug)]
pub struct SystemResources {
    sys: System, // Store the System object to refresh data
    pub cpu_core_count: usize,
    pub cpu_load_avg_one: f32,    // 1-minute load average
    pub cpu_load_avg_five: f32,   // 5-minute load average
    pub cpu_load_avg_fifteen: f32,// 15-minute load average
    pub ram_total_gb: f32,
    pub ram_used_gb: f32,
    pub ram_available_gb: f32, // Calculated: total - used (sysinfo provides used, not free directly for all OS)
    pub swap_total_gb: f32,
    pub swap_used_gb: f32,
    pub swap_available_gb: f32, // Calculated

    // Placeholders for future, more complex metrics
    pub vram_available_gb: Option<f32>,
    pub ssd_io_bandwidth_mbps: Option<f32>, // MB/s
}

impl SystemResources {
    pub fn new() -> Self {
        let mut sys = System::new_all(); // new_all() gets all basic info
        sys.refresh_all(); // Refresh to get initial values for CPU, memory, etc.
        
        // For load average, sysinfo provides it as an array [1min, 5min, 15min]
        let load_avg = sys.load_average();

        let total_memory_kib = sys.total_memory();
        let used_memory_kib = sys.used_memory(); // sysinfo reports KiB for memory values.
                                                 // "available_memory" gives a clearer sense of free memory across OSes.
        let available_memory_kib = sys.available_memory();


        Self {
            sys, // Keep sys to allow refreshing later
            cpu_core_count: sys.cpus().len(), // Number of logical cores
            cpu_load_avg_one: load_avg.one as f32,
            cpu_load_avg_five: load_avg.five as f32,
            cpu_load_avg_fifteen: load_avg.fifteen as f32,
            ram_total_gb: total_memory_kib as f32 / KIB_TO_GB,
            ram_used_gb: used_memory_kib as f32 / KIB_TO_GB,
            ram_available_gb: available_memory_kib as f32 / KIB_TO_GB,
            swap_total_gb: sys.total_swap() as f32 / KIB_TO_GB,
            swap_used_gb: sys.used_swap() as f32 / KIB_TO_GB,
            swap_available_gb: (sys.total_swap() - sys.used_swap()) as f32 / KIB_TO_GB,
            vram_available_gb: None, // Placeholder
            ssd_io_bandwidth_mbps: None, // Placeholder
        }
    }

    pub fn refresh(&mut self) {
        self.sys.refresh_memory(); // More specific refresh for memory
        self.sys.refresh_cpu();    // For CPU load, though load_average is system-wide not just CPU usage.
                                   // sysinfo's CPU usage is per-core instantaneous, load_average is more traditional.

        let load_avg = self.sys.load_average();
        self.cpu_load_avg_one = load_avg.one as f32;
        self.cpu_load_avg_five = load_avg.five as f32;
        self.cpu_load_avg_fifteen = load_avg.fifteen as f32;

        self.ram_used_gb = self.sys.used_memory() as f32 / KIB_TO_GB;
        self.ram_available_gb = self.sys.available_memory() as f32 / KIB_TO_GB;
        self.swap_used_gb = self.sys.used_swap() as f32 / KIB_TO_GB;
        self.swap_available_gb = (self.sys.total_swap() - self.sys.used_swap()) as f32 / KIB_TO_GB;
        
        // Individual CPU usage can be fetched via self.sys.cpus() list if needed.
        // For example, overall CPU usage:
        // let overall_cpu_usage = self.sys.global_cpu_info().cpu_usage();
        // This could be another field if instantaneous global CPU % is preferred over load average.
    }
}

// Default implementation for cases where new() might not be suitable directly
impl Default for SystemResources {
    fn default() -> Self {
        Self::new()
    }
}
