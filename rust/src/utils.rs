use std::time::Instant;

pub fn display_time<T: Into<u128>>(micro_seconds: T) -> String {
    let mut v: u128 = micro_seconds.into();

    let micros = v % 1000;
    v /= 1000;
    let mills = v % 1000;
    v /= 1000;
    let seconds = v % 60;
    v /= 60;
    let minutes = v % 60;
    v /= 60;
    let hours = v;

    if hours != 0 {
        format!("{hours}h{minutes}m{seconds}s+{mills}ms")
    } else if minutes != 0 {
        format!("{minutes}m{seconds}s+{mills}ms")
    } else if seconds != 0 {
        format!("{seconds}s+{mills}ms")
    } else if mills != 0 {
        format!("{mills}ms+{micros}us")
    } else {
        format!("{micros}us")
    }
}

pub fn calculate_cost_time<F, T>(behaviour: &str, f: F) -> (T, u128)
where
    F: FnOnce() -> T,
{
    let start_time = Instant::now();
    let result = f();
    let elapsed = start_time.elapsed();

    let time = display_time(elapsed.as_micros());
    println!("[{behaviour}] runs in {time}");

    (result, elapsed.as_micros())
}

pub mod timing {
    use std::sync::atomic::AtomicU64;
    use std::time::{SystemTime, UNIX_EPOCH};

    pub static GLOBAL_TIME_POINT: AtomicU64 = AtomicU64::new(0);
    pub static mut GLOBAL_TIME_SLOTS: [u64; 10] = [0u64; 10];

    pub fn current_timestamp() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64
    }

    #[macro_export]
    macro_rules! reset_time_point {
        () => {
            let now = crate::utils::timing::current_timestamp();
            crate::utils::timing::GLOBAL_TIME_POINT.store(now, std::sync::atomic::Ordering::Relaxed);
        };
    }

    #[macro_export]
    macro_rules! calc_elapsed_time {
        () => {{
            let now = crate::utils::timing::current_timestamp();
            let last = crate::utils::timing::GLOBAL_TIME_POINT.swap(now, std::sync::atomic::Ordering::Relaxed);
            (now - last) as u64
        }};
        (msg = $msg: expr) => {{
            let delta = calc_elapsed_time!();
            println!("{}, elapsed time: {}", $msg, super::display_time(delta));
            delta
        }};
        (slot = $slot: expr) => {{
            use std::ptr::addr_of_mut;
            let delta = calc_elapsed_time!();

            unsafe {
                let slots = addr_of_mut!(crate::utils::timing::GLOBAL_TIME_SLOTS);
                let len = (*slots).len();

                if $slot >= len {
                    panic!("slot index out of range: {}/{}", $slot, len);
                }

                (*slots)[$slot] += delta;
            }
        }};
        ($slot: expr, $msg: expr) => {{
            use std::ptr::addr_of_mut;
            let delta = calc_elapsed_time!(msg = $msg);

            unsafe {
                let slots = addr_of_mut!(crate::utils::timing::GLOBAL_TIME_SLOTS);
                let len = (*slots).len();

                if $slot >= len {
                    panic!("slot index out of range: {}/{}", $slot, len);
                }

                (*slots)[$slot] += delta;
            }
        }}
    }

    #[macro_export]
    macro_rules! get_slots {
        ($n: expr) => {{
            use bap::utils::timing;
            unsafe {
                timing::GLOBAL_TIME_SLOTS.iter().take($n)
            }
        }};
    }

    #[macro_export]
    macro_rules! reset_slots {
        () => {{
            use bap::utils::timing;
            unsafe {
                timing::GLOBAL_TIME_SLOTS.fill(0);
            }
        }};
    }
}

#[macro_export]
macro_rules! set {
    () => {{
        use ahash::HashSet;
        HashSet::new()
    }};
    ($e: expr) => {{
            use ahash::HashSet;
            let mut set = HashSet::new();
            set.insert($e);
            set
    }};
    ($($e: expr), *) => {{
        use ahash::HashSet;
        let mut set = HashSet::new();
        $(set.insert($e);)*
        set
    }}
}
