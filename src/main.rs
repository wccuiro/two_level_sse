use ndarray::{array, Array1, Array2, Axis};
use ndarray::linalg::kron;

use num_complex::Complex64;
use rand::Rng;

use rayon::prelude::*;
use plotters::prelude::*;


use std::fs::{self, File, create_dir_all, remove_file};
use std::io::Write;
use std::path::Path;
// use std::sync::atomic::{AtomicUsize, Ordering};



use csv::ReaderBuilder;


fn gen_sigma_x() -> Array2<Complex64> {
    array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ]
}


fn gen_sigma_y() -> Array2<Complex64> {
    array![
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
        [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)]
        ]
}
    
fn gen_sigma_z() -> Array2<Complex64> {
    array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
    ]
}

fn simulate_spin_jump_cm(
    omega: f64,
    gamma: f64,
    dt: f64,
    total_time: f64,
) -> (Vec<f64>, Vec<f64>) {

    let sigma_x = gen_sigma_x();

    let sigma_z = gen_sigma_z();

    let sigma_plus = gen_sigma_x().mapv(|e| e * Complex64::new(0.5, 0.0)) + gen_sigma_y().mapv(|e| e * Complex64::new(0.0, 0.5)) ;

    let sigma_minus = gen_sigma_x().mapv(|e| e * Complex64::new(0.5, 0.0)) - gen_sigma_y().mapv(|e| e * Complex64::new(0.0, 0.5)) ;

    let steps = (total_time / dt).ceil() as usize;

    let mut psi: Array1<Complex64>  = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let mut sz_exp = Vec::with_capacity(steps);

    let mut time_jumps:Vec<f64> = Vec::new();

    let sigma_pm = sigma_plus.dot(&sigma_minus);
    let h_eff = sigma_x.mapv(|e| e * omega) - sigma_pm.mapv(|e| e * (gamma * Complex64::new(0.0, 0.5)));

    let mut rng = rand::thread_rng();

    for i in 0..steps {
        let amp = psi.mapv(|x| x.conj()).dot(&sigma_pm.dot(&psi)).re;
        let p_jump = gamma * amp * dt;
        let dpsi_nh = (&psi.mapv(|e| e * (gamma * amp * 0.5))
            - &(h_eff.dot(&psi).mapv(|e| Complex64::new(0.0, 1.0) * e)))
            .mapv(|e| e * dt);

        if rng.gen::<f64>() <= p_jump {
            let norm_factor = amp.sqrt();
            psi = sigma_minus.dot(&psi).mapv(|e| e * (1. / norm_factor));
            time_jumps.push(i as f64 * dt); 
        }

        psi = &psi + &dpsi_nh;
        let norm = psi.mapv(|x| x.conj()).dot(&psi).re.sqrt();
        psi = psi.mapv(|e| e / norm);

        let szz = psi.mapv(|x| x.conj()).dot(&sigma_z.dot(&psi)).re;
        sz_exp.push(szz);
    }
    // println!("Number of jumps cm: {}",time_jumps.len());

    (sz_exp, time_jumps)
}

fn simulate_spin_jump_rj(
    omega: f64,
    gamma: f64,
    dt: f64,
    total_time: f64,
) -> (Vec<f64>, Vec<f64>) {
    let max_steps = (total_time / dt).ceil() as usize;
    let mut sz_exp = Vec::with_capacity(max_steps);

    let mut time_jumps:Vec<f64> = Vec::new();

    let sigma_x = gen_sigma_x();

    let sigma_z = gen_sigma_z();

    let sigma_plus = gen_sigma_x().mapv(|e| e * Complex64::new(0.5, 0.0)) + gen_sigma_y().mapv(|e| e * Complex64::new(0.0, 0.5)) ;

    let sigma_minus = gen_sigma_x().mapv(|e| e * Complex64::new(0.5, 0.0)) - gen_sigma_y().mapv(|e| e * Complex64::new(0.0, 0.5)) ;

    let sigma_pm = sigma_plus.dot(&sigma_minus);
    let h_eff = sigma_x.mapv(|e| e * omega) - sigma_pm.mapv(|e| e * (gamma * Complex64::new(0.0, 0.5)));

    let mut psi = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let mut rng = rand::thread_rng();

    let mut p0 = 1.0;
    let mut r = rng.gen::<f64>();
    let mut nh_evol = true;

    for i in 0..max_steps {
        if nh_evol {
            r = rng.gen();
            p0 = 1.0;
            nh_evol = false;
        }

        let amp = psi.mapv(|x| x.conj()).dot(&sigma_pm.dot(&psi)).re;
        let dpsi_nh = (&psi.mapv(|e| e * (gamma * amp * 0.5))
            - &(h_eff.dot(&psi).mapv(|e| Complex64::new(0.0, 1.0) * e)))
            .mapv(|e| e * dt);

        if r >= p0 {
            let norm_factor = amp.sqrt();
            psi = sigma_minus.dot(&psi).mapv(|e| e * (1. / norm_factor));
            time_jumps.push(i as f64 * dt); 
            nh_evol = true;
        }

        psi = &psi + &dpsi_nh;
        let norm = psi.mapv(|x| x.conj()).dot(&psi).re.sqrt();
        psi.mapv_inplace(|e| e / norm);

        let szz = psi.mapv(|x| x.conj()).dot(&sigma_z.dot(&psi)).re;
        sz_exp.push(szz);

        let p_jump = gamma * amp * dt;
        p0 *= 1.0 - p_jump;
    }

    // println!("Number of jumps rj: {}",time_jumps.len());


    (sz_exp, time_jumps)
}


fn lindblad_simulation(
    omega: f64,
    gamma: f64,
    dt: f64,
    total_time: f64,
) -> Vec<f64> {
    let max_steps = (total_time / dt).ceil() as usize;
    let mut sz_exp = Vec::with_capacity(max_steps);

    let sigma_x = gen_sigma_x();

    let sigma_z = gen_sigma_z();

    let sigma_plus = gen_sigma_x().mapv(|e| e * Complex64::new(0.5, 0.0)) + gen_sigma_y().mapv(|e| e * Complex64::new(0.0, 0.5)) ;

    let sigma_minus = gen_sigma_x().mapv(|e| e * Complex64::new(0.5, 0.0)) - gen_sigma_y().mapv(|e| e * Complex64::new(0.0, 0.5)) ;

    // let sigma_pm = sigma_plus.dot(&sigma_minus);

    let h_eff = sigma_x.mapv(|e| e * omega);
    let identity = array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
    ];
    
    let v_sigmaz = kron(&sigma_z, &identity);
    let v_identity = array![
        Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)
    ];

    let commutator = kron(&h_eff, &identity) - kron(&identity, &h_eff.t());
    let term2 = kron(&sigma_minus, &sigma_plus.t());

    let left = sigma_plus.dot(&sigma_minus);
    let right = sigma_minus.t().dot(&sigma_plus.t());

    let anticommutator = (kron(&left, &identity) + kron(&identity, &right)).mapv(|e| e * 0.5);

    let s_l = &commutator.mapv(|e| e * Complex64::new(0.0, -1.0)) + (&term2 - &anticommutator).mapv(|e| e * gamma);

    let mut rho = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];

    for _ in 0..max_steps {
        rho = &rho + (&s_l.dot(&rho)).mapv(|e| e * dt);
        let norm = v_identity.dot(&rho).re;
        rho = rho.mapv(|e| e / norm);

        let szz = v_identity.dot(&v_sigmaz.dot(&rho)).re;
        sz_exp.push(szz);
    }

    sz_exp
}

fn bin_width(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 1.0; // Default bin size if no data
    }

    fn quantile(data: &[f64], prob: f64) -> f64 {
        let n = data.len();
        let idx = prob * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        if lo == hi {
            data[lo]
        } else {
            let frac = idx - lo as f64;
            data[lo] * (1.0 - frac) + data[hi] * frac
        }
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q1 = quantile(&sorted, 0.25);
    let q3 = quantile(&sorted, 0.75);
    let iqr = q3 - q1;
    let n = data.len() as f64;

    (2.0 * iqr) / n.cbrt()
}    

fn counts_per_bin(
    data: &[f64],
    bin_width: f64,
    min: f64,
    max: f64,
) -> Vec<usize> {
    let num_bins = ((max - min) / bin_width).ceil() as usize;
    let mut counts = vec![0; num_bins];

    for &value in data {
        if value >= min && value < max {
            let bin_index = ((value - min) / bin_width).floor() as usize;
            if bin_index < num_bins {
                counts[bin_index] += 1;
            }
        }
    }

    counts
}

fn compute_tick_times(times: &Vec<f64>, m: usize) -> Array1<f64> {
    let ticks: Vec<f64> = times
        .iter()
        .enumerate()
        .filter_map(|(idx, &t)| {
            // idx starts at 0, so (idx+1)%m == 0 picks the mᵗʰ, 2mᵗʰ, ...
            if (idx + 1) % m == 0 {
                Some(t)
            } else {
                None
            }
        })
        .collect();
    Array1::from(ticks)
}

/// Given an Array1<f64> of tick times [t₁, t₂, …], return Vec<f64> of [t₂−t₁, t₃−t₂, …]
fn analyze_waiting_times(ticks: &Array1<f64>) -> Vec<f64> {
    let slice = ticks
        .as_slice()
        .expect("tick array must be contiguous");
    slice
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect()
}



fn plot_histogram(
    counts: &Vec<usize>,
    bin_width: f64,
    min: f64,
    max: f64,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {

    // Set up drawing area
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_count = *counts.iter().max().unwrap_or(&0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Histogram", ("FiraCode Nerd Font", 40))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min..max, 0..max_count)?;

    chart.configure_mesh()
        .x_desc("Value")
        .y_desc("Count")
        .draw()?;

    // Draw bars
    for (i, &count) in counts.iter().enumerate() {
        let x0 = min + i as f64 * bin_width;
        let x1 = x0 + bin_width;
        chart.draw_series(
            std::iter::once(Rectangle::new(
                [(x0, 0), (x1, count)],
                BLUE.filled(),
            )),
        )?;
    }

    Ok(())
}

fn plot_trajectory_avg(
    avg_cm: Array1<f64>,
    avg_rj: Array1<f64>,
    lindblad_avg: Vec<f64>,
    steps: usize,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {

    let root = BitMapBackend::new(&filename, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Average <σ_z> trajectory", ("FiraCode Nerd Font", 30))
        .margin(100)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..steps, -1.0..1.0)?;

    chart.configure_mesh()
        .x_desc("Time steps")
        .y_desc("<σ_z>")
        .label_style(("FiraCode Nerd Font", 30).into_font())
        .draw()?;

    chart.draw_series(LineSeries::new(
        avg_cm.iter().enumerate().map(|(x, y)| (x, *y)),
        &BLUE,
    ))?
    .label("CM")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.draw_series(LineSeries::new(
        avg_rj.iter().enumerate().map(|(x, y)| (x, *y)),
        &RED,
    ))?
    .label("RJ")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.draw_series(LineSeries::new(
        lindblad_avg.iter().enumerate().map(|(x, y)| (x, *y)),
        &MAGENTA,
    ))?
    .label("Avg")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));

    chart.configure_series_labels()
    .position(SeriesLabelPosition::UpperRight)
    .label_font(("FiraCode Nerd Font", 40).into_font())
    .draw()?;

    Ok(())
}


fn clean_directory(dir: &str) -> std::io::Result<()> {
    if Path::new(dir).exists() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                remove_file(path)?;
            }
        }
    }
    Ok(())
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simulation parameters
    let omega: f64 = 10.0;
    let gamma: f64 = 7.0;
    let dt: f64 = 0.001;
    let total_time: f64 = 30.0;
    let num_trajectories: usize = 5000;
    let m: usize = 10;

    // Precompute number of steps for reshaping
    // let steps: usize = (total_time / dt).ceil() as usize;

    // Prepare output directories
    // let base_dir = "trajectories";
    // let cm_dir = format!("{}/cm", base_dir);
    // let rj_dir = format!("{}/rj", base_dir);


    // // 1) Clean out old files (if any)
    // clean_directory(&cm_dir)?;
    // clean_directory(&rj_dir)?;
    // // 2) (Re)create the directories
    // create_dir_all(&cm_dir)?;
    // create_dir_all(&rj_dir)?;

    // // Atomic counters for unique filenames
    // let counter_cm = AtomicUsize::new(0);
    // let counter_rj = AtomicUsize::new(0);

    // chunking
    let chunk_size: usize = 500;
    let cm_chunks = (num_trajectories + chunk_size - 1) / chunk_size;
    let rj_chunks = cm_chunks;

    // precompute steps
    let steps: usize = (total_time / dt).ceil() as usize;

    // ─── Prepare directories ──────────────────────────────────────────────────
    let base_dir = "trajectories";
    let cm_dir   = format!("{}/cm", base_dir);
    let rj_dir   = format!("{}/rj", base_dir);

    // clean & recreate
    clean_directory(&cm_dir)?;
    clean_directory(&rj_dir)?;
    
    create_dir_all(&cm_dir)?;
    create_dir_all(&rj_dir)?;

    // ─── CM: simulate & write in chunks ───────────────────────────────────────
    (0..cm_chunks).into_par_iter().for_each(|chunk_idx| {
        let start = chunk_idx * chunk_size;
        let end   = ((chunk_idx + 1) * chunk_size).min(num_trajectories);
        let count = end - start;

        let mut traj_buf  = Vec::with_capacity(count);
        let mut jumps_buf = Vec::with_capacity(count);
        for _ in 0..count {
            let (traj, jtimes) = simulate_spin_jump_cm(omega, gamma, dt, total_time);
            traj_buf.push(traj);
            jumps_buf.push(jtimes);
        }

        // write trajectories
        let path_t = format!("{}/trajectory_chunk_{}.csv", cm_dir, chunk_idx);
        let mut ft = File::create(&path_t).expect("CM traj chunk create failed");
        writeln!(ft, "traj_id,time,state").unwrap();
        for (local_id, traj) in traj_buf.iter().enumerate() {
            let gid = start + local_id;
            for (i, &state) in traj.iter().enumerate() {
                let t = i as f64 * dt;
                writeln!(ft, "{},{:.6},{:.6}", gid, t, state).unwrap();
            }
        }

        // write jumps
        let path_j = format!("{}/jumps_chunk_{}.csv", cm_dir, chunk_idx);
        let mut fj = File::create(&path_j).expect("CM jumps chunk create failed");
        writeln!(fj, "traj_id,jump_time").unwrap();
        for (local_id, times) in jumps_buf.iter().enumerate() {
            let gid = start + local_id;
            for &t in times.iter() {
                writeln!(fj, "{},{}", gid, t).unwrap();
            }
        }
    });

    // ─── RJ: simulate & write in chunks ───────────────────────────────────────
    (0..rj_chunks).into_par_iter().for_each(|chunk_idx| {
        let start = chunk_idx * chunk_size;
        let end   = ((chunk_idx + 1) * chunk_size).min(num_trajectories);
        let count = end - start;

        let mut traj_buf  = Vec::with_capacity(count);
        let mut jumps_buf = Vec::with_capacity(count);
        for _ in 0..count {
            let (traj, jtimes) = simulate_spin_jump_rj(omega, gamma, dt, total_time);
            traj_buf.push(traj);
            jumps_buf.push(jtimes);
        }

        // write trajectories
        let path_t = format!("{}/trajectory_chunk_{}.csv", rj_dir, chunk_idx);
        let mut ft = File::create(&path_t).expect("RJ traj chunk create failed");
        writeln!(ft, "traj_id,time,state").unwrap();
        for (local_id, traj) in traj_buf.iter().enumerate() {
            let gid = start + local_id;
            for (i, &state) in traj.iter().enumerate() {
                let t = i as f64 * dt;
                writeln!(ft, "{},{:.6},{:.6}", gid, t, state).unwrap();
            }
        }

        // write jumps
        let path_j = format!("{}/jumps_chunk_{}.csv", rj_dir, chunk_idx);
        let mut fj = File::create(&path_j).expect("RJ jumps chunk create failed");
        writeln!(fj, "traj_id,jump_time").unwrap();
        for (local_id, times) in jumps_buf.iter().enumerate() {
            let gid = start + local_id;
            for &t in times.iter() {
                writeln!(fj, "{},{}", gid, t).unwrap();
            }
        }
    });

    // ─── Read back CM data from chunks ────────────────────────────────────────
    let mut trajectories_cm = vec![Vec::new(); num_trajectories];
    let mut times_jumps_cm  = vec![Vec::new(); num_trajectories];

    for chunk_idx in 0..cm_chunks {
        // trajectories
        let path = format!("{}/trajectory_chunk_{}.csv", cm_dir, chunk_idx);
        let mut rdr = ReaderBuilder::new().has_headers(true).from_path(&path)?;
        for rec in rdr.records() {
            let row = rec?;
            let id: usize = row[0].parse()?;
            let state: f64 = row[2].parse()?;
            trajectories_cm[id].push(state);
        }
        // jumps
        let path = format!("{}/jumps_chunk_{}.csv", cm_dir, chunk_idx);
        let mut rdr = ReaderBuilder::new().has_headers(true).from_path(&path)?;
        for rec in rdr.records() {
            let row = rec?;
            let id: usize = row[0].parse()?;
            let t:  f64   = row[1].parse()?;
            times_jumps_cm[id].push(t);
        }
    }

    // ─── Read back RJ data from chunks ────────────────────────────────────────
    let mut trajectories_rj = vec![Vec::new(); num_trajectories];
    let mut times_jumps_rj  = vec![Vec::new(); num_trajectories];

    for chunk_idx in 0..rj_chunks {
        // trajectories
        let path = format!("{}/trajectory_chunk_{}.csv", rj_dir, chunk_idx);
        let mut rdr = ReaderBuilder::new().has_headers(true).from_path(&path)?;
        for rec in rdr.records() {
            let row = rec?;
            let id: usize = row[0].parse()?;
            let state: f64 = row[2].parse()?;
            trajectories_rj[id].push(state);
        }
        // jumps
        let path = format!("{}/jumps_chunk_{}.csv", rj_dir, chunk_idx);
        let mut rdr = ReaderBuilder::new().has_headers(true).from_path(&path)?;
        for rec in rdr.records() {
            let row = rec?;
            let id: usize = row[0].parse()?;
            let t:  f64   = row[1].parse()?;
            times_jumps_rj[id].push(t);
        }
    }

    // Lindblad average trajectory
    let lindblad_avg: Vec<f64> = lindblad_simulation(omega, gamma, dt, total_time);

    // Compute waiting times and flatten & sort
    let mut flat_waits_cm: Vec<f64> = times_jumps_cm
        .par_iter()
        .flat_map(|times| analyze_waiting_times(&compute_tick_times(times, m)))
        .collect();
    flat_waits_cm.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let average_cm = flat_waits_cm.iter().copied().sum::<f64>() / flat_waits_cm.len() as f64;
    let variance_cm = flat_waits_cm.iter().map(|x| (x - average_cm).powi(2)).sum::<f64>() / flat_waits_cm.len() as f64;

    let accuracy_cm: f64 = average_cm.powi(2) / variance_cm;
    let resolution_cm: f64 = 1.0 / average_cm;

    println!("CM Accuracy: {}, Resolution: {}", accuracy_cm, resolution_cm);

    let mut flat_waits_rj: Vec<f64> = times_jumps_rj
        .par_iter()
        .flat_map(|times| analyze_waiting_times(&compute_tick_times(times, m)))
        .collect();
    flat_waits_rj.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let average_rj = flat_waits_rj.iter().copied().sum::<f64>() / flat_waits_rj.len() as f64;
    let variance_rj = flat_waits_rj.iter().map(|x| (x - average_rj).powi(2)).sum::<f64>() / flat_waits_rj.len() as f64;

    let accuracy_rj: f64 = average_rj.powi(2) / variance_rj;
    let resolution_rj: f64 = 1.0 / average_rj;

    println!("CM Accuracy: {}, Resolution: {}", accuracy_rj, resolution_rj);

    // Histogram binning
    let bin_width_cm = bin_width(&flat_waits_cm);
    let bin_width_rj = bin_width(&flat_waits_rj);

    let counts_cm = counts_per_bin(&flat_waits_cm, bin_width_cm, 0.0, total_time);
    let counts_rj = counts_per_bin(&flat_waits_rj, bin_width_rj, 0.0, total_time);

    // Plot histograms
    let filename_cm = format!(
        "histogram_cm_omega-{}_gamma-{}_dt-{}_ntraj-{}.png",
        omega, gamma, dt, num_trajectories
    );
    plot_histogram(&counts_cm, bin_width_cm, 0.0, total_time, &filename_cm)?;

    let filename_rj = format!(
        "histogram_rj_omega-{}_gamma-{}_dt-{}_ntraj-{}.png",
        omega, gamma, dt, num_trajectories
    );
    plot_histogram(&counts_rj, bin_width_rj, 0.0, total_time, &filename_rj)?;

    // Compute average trajectories
    let flat_cm: Vec<f64> = trajectories_cm.into_iter().flatten().collect();
    let data_cm = Array2::from_shape_vec((num_trajectories, steps), flat_cm)?;

    let flat_rj: Vec<f64> = trajectories_rj.into_iter().flatten().collect();
    let data_rj = Array2::from_shape_vec((num_trajectories, steps), flat_rj)?;

    let avg_cm: Array1<f64> = data_cm.mean_axis(Axis(0)).unwrap();
    let avg_rj: Array1<f64> = data_rj.mean_axis(Axis(0)).unwrap();

    // Plot average trajectories
    let filename = format!(
        "plot_omega-{}_gamma-{}_dt-{}_ntraj-{}.png",
        omega, gamma, dt, num_trajectories
    );
    plot_trajectory_avg(avg_cm, avg_rj, lindblad_avg, steps, &filename)?;

    println!("Simulation completed successfully!");
    println!("Generated files: {}, {}, {}", filename_cm, filename_rj, filename);
    Ok(())
}



// fn main() -> Result<(), Box<dyn std::error::Error>>{
//     let omega: f64 = 1.7;
//     let gamma: f64 = 0.1;
//     let dt: f64 = 0.001;
//     let total_time: f64 = 30.0;
//     let num_trajectories: usize = 5000;

//     let m: usize = 5;

//     // Calculate number of steps as usize
//     let steps: usize = (total_time / dt).ceil() as usize;

//     // Run simulations in parallel, passing total_time
//     let (trajectories_cm, times_jumps_cm): (Vec<Vec<f64>> , Vec<Vec<f64>>) = (0..num_trajectories)
//         .into_par_iter()
//         .map(|_| simulate_spin_jump_cm(omega, gamma, dt, total_time))
//         .unzip();

//     // Run simulations in parallel, passing total_time
//     let (trajectories_rj, times_jumps_rj): (Vec<Vec<f64>> , Vec<Vec<f64>>) = (0..num_trajectories)
//         .into_par_iter()
//         .map(|_| simulate_spin_jump_rj(omega, gamma, dt, total_time))
//         .unzip();
    
//     let lindblad_avg: Vec<f64> = lindblad_simulation(omega, gamma, dt, total_time);

//     let all_waiting_times_rj: Vec<Vec<f64>> = times_jumps_rj
//         .par_iter()                     // Rayon parallel iterator over &Vec<f64>
//         .map(|times: &Vec<f64>| {
//             let ticks = compute_tick_times(times, m);
//             analyze_waiting_times(&ticks)
//         })
//         .collect();

//     // Optionally flatten into one big Vec<f64>:
//     let mut flat_waits_rj: Vec<f64> = all_waiting_times_rj
//         .into_par_iter()
//         .flatten()
//         .collect();

//     flat_waits_rj.sort_by(|a, b| a.partial_cmp(b).unwrap());

//     let all_waiting_times_cm: Vec<Vec<f64>> = times_jumps_cm
//         .par_iter()                     // Rayon parallel iterator over &Vec<f64>
//         .map(|times: &Vec<f64>| {
//             let ticks = compute_tick_times(times, m);
//             analyze_waiting_times(&ticks)
//         })
//         .collect();

//     // Optionally flatten into one big Vec<f64>:
//     let mut flat_waits_cm: Vec<f64> = all_waiting_times_cm
//         .into_par_iter()
//         .flatten()
//         .collect();

//     flat_waits_cm.sort_by(|a, b| a.partial_cmp(b).unwrap());

//     let bin_width_cm = bin_width(&flat_waits_cm);
//     let bin_width_rj = bin_width(&flat_waits_rj);


//     let counts_cm = counts_per_bin(
//         &flat_waits_cm,
//         bin_width_cm,
//         0.0,
//         total_time,
//     );

//     let counts_rj = counts_per_bin(
//         &flat_waits_rj,
//         bin_width_rj,
//         0.0,
//         total_time,
//     );

//     let filename_cm = format!("histogram_cm_omega-{}_gamma-{}_dt-{}_ntraj-{}.png", omega, gamma, dt, num_trajectories);
//     plot_histogram(&counts_cm, bin_width_cm, 0.0, total_time, &filename_cm)?;

//     let filename_rj = format!("histogram_rj_omega-{}_gamma-{}_dt-{}_ntraj-{}.png", omega, gamma, dt, num_trajectories);
//     plot_histogram(&counts_rj, bin_width_rj, 0.0, total_time, &filename_rj)?;


//     // Flatten and reshape into 2D array
//     let flat_cm: Vec<f64> = trajectories_cm.into_iter().flatten().collect();
//     let data_cm = Array2::from_shape_vec((num_trajectories, steps), flat_cm)?;

//     let flat_rj: Vec<f64> = trajectories_rj.into_iter().flatten().collect();
//     let data_rj = Array2::from_shape_vec((num_trajectories, steps), flat_rj)?;

//     // Mean over rows (trajectories)
//     let avg_cm: Array1<f64> = data_cm.mean_axis(Axis(0)).unwrap();

//     let avg_rj: Array1<f64> = data_rj.mean_axis(Axis(0)).unwrap();

//     // Plot the average trajectory
//     let filename = format!("plot_omega-{}_gamma-{}_dt-{}_ntraj-{}.png", omega, gamma, dt, num_trajectories);
//     plot_trajectory_avg(avg_cm, avg_rj, lindblad_avg, steps, &filename)?;

//     println!("Simulation completed successfully!");
//     println!("Generated files: {}, {}, {}", filename_cm, filename_rj, filename);
//     Ok(())
// }
