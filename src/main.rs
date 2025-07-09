use ndarray::{array, Array1, Array2, Axis};
use ndarray::linalg::kron;

use num_complex::Complex64;
use rand::Rng;

use rayon::prelude::*;
use plotters::prelude::*;


// use std::fs::{self, File, create_dir_all, remove_file};
// use std::io::Write;
// use std::path::Path;
// use std::sync::atomic::{AtomicUsize, Ordering};

use indicatif::{ProgressBar, ProgressStyle};
// use std::sync::Arc;
// use std::time::Duration;


// use csv::ReaderBuilder;


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

fn steady_state(gamma: f64, omega: f64) -> (Array2<Complex64>, Array1<Complex64>, Array1<Complex64>, Array1<f64>) {
    let delta: f64 =  (gamma.powi(4) + 16. * gamma.powi(2) * omega.powi(2)).sqrt();

    // Eigenvalues
    let eig_val_1 = (gamma.powi(2) + 8. * omega.powi(2) - delta) / (2. * gamma.powi(2) + 8. * omega.powi(2));
    
    let eig_val_2 = (gamma.powi(2) + 8. * omega.powi(2) + delta) / (2. * gamma.powi(2) + 8. * omega.powi(2));

    let mut eig_vals: Array1<f64> = array![eig_val_1, eig_val_2];
    eig_vals = eig_vals.mapv(|e| e/ (eig_val_1 + eig_val_2));

    // Eigenvectors
    let i = Complex64::new(0.0, 1.0);

    // psi1
    let top11 = i * (gamma + (gamma.powi(2) + 16. * omega.powi(2)).sqrt());
    let norm1 = (16.0 * omega.powi(2) + (gamma - (gamma.powi(2) + 16. * omega.powi(2)).sqrt()).powi(2)).sqrt();
    let top12 = 4. * omega;
    let mut psi1 = Array1::from(vec![top11 / norm1, Complex64::new( top12 / norm1, 0.0)]);
    psi1 /= psi1.mapv(|e| e.conj()).dot(&psi1).sqrt();

    // psi2
    let top21 = i * (gamma - (gamma.powi(2) + 16. * omega.powi(2)).sqrt());
    let norm2 = (16.0 * omega.powi(2) + (gamma + (gamma.powi(2) + 16. * omega.powi(2)).sqrt()).powi(2)).sqrt();
    let top22 = 4. * omega;
    let mut psi2 = Array1::from(vec![top21 / norm2, Complex64::new( top22 / norm2, 0.0)]);
    psi2 /= psi2.mapv(|e| e.conj()).dot(&psi2).sqrt();


    // Steady state density matrix
    // Compute |psi1><psi1|
    let a = psi1[0];
    let b = psi1[1];
    let op1 = array![
        [a * a.conj(), a * b.conj()],
        [b * a.conj(), b * b.conj()]
    ];

    // Compute |psi2><psi2|
    let c = psi2[0];
    let d = psi2[1];
    let op2 = array![
        [c * c.conj(), c * d.conj()],
        [d * c.conj(), d * d.conj()]
    ];

    // Weighted sum
    let mut pi = op1.mapv(|v| v * eig_vals[0]) + op2.mapv(|v| v * eig_vals[1]);
    // let pi_def:Array2<Complex64> = array![
    //     [Complex64::new(4. * omega.powi(2), 0.0), Complex64::new(0.0, 2. * omega * gamma)],
    //     [Complex64::new(0.0, -2. * omega * gamma), Complex64::new(gamma.powi(2) + 4. * omega.powi(2), 0.0)]
    // ]/ Complex64::new( (gamma.powi(2) + 8. * omega.powi(2)) ,0.0);

    // let pi_dif = pi.clone() - pi_def;
    // println!("Difference between steady state and analytical solution: \n{:?}", pi_dif);

    // Normalize by trace
    let trace: f64 = (pi[(0, 0)] + pi[(1, 1)]).re;
    // println!("Trace of steady state density matrix: {}", trace);
    pi /= Complex64::new(trace, 0.0);

    (pi, psi1, psi2, eig_vals)
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

    // Initialize the state vector psi
    let (_, psi1, psi2, eigvals) = steady_state(gamma, omega);
    let mut rng = rand::thread_rng();
    let i = if rng.gen::<f64>() < eigvals[0] { 0 } else { 1 };
    let mut psi;
        
    if i == 0 {
        psi = psi1;
        psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();
    } else {
        psi = psi2;
        psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();
    }

    let mut sz_exp = Vec::with_capacity(steps);

    let mut time_jumps:Vec<f64> = Vec::new();

    let sigma_pm = sigma_plus.dot(&sigma_minus);
    let h_eff = sigma_x.mapv(|e| e * omega) - sigma_pm.mapv(|e| e * (gamma * Complex64::new(0.0, 0.5)));

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

    // println!("CM simulation completed with {} jumps.", time_jumps.len());

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

    // Initialize the state vector psi
    let (_, psi1, psi2, eigvals) = steady_state(gamma, omega);
    let mut rng = rand::thread_rng();
    let i = if rng.gen::<f64>() < eigvals[0] { 0 } else { 1 };
    let mut psi;
        
    if i == 0 {
        psi = psi1;
        psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();
    } else {
        psi = psi2;
        psi /= psi.mapv(|e| e.conj()).dot(&psi).sqrt();
    }

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

    // println!("CM simulation completed with {} jumps.", time_jumps.len());

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

    // Initialize the state vector psi
    let (pi, _, _, _) = steady_state(gamma, omega);
    // println!("{:?}",pi);
    let mut rho: Array1<Complex64> = pi.into_raw_vec().into();
    // println!("{:?}",rho);
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
) -> Vec<f64> {
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

    let total: f64 = counts.iter().sum::<usize>() as f64;
    let norm_counts: Vec<f64> = counts.iter().map(|&e| e as f64 / total).collect();

    norm_counts
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
    counts: &Vec<f64>,
    bin_width: f64,
    min: f64,
    max: f64,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {

    // Set up drawing area
    let root = BitMapBackend::new(filename, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_count = counts.iter().cloned().fold(0.0_f64, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Histogram", ("FiraCode Nerd Font", 40))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min..max, 0.0..max_count)?;

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
                [(x0, 0.0), (x1, count)],
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
    min:f64,
    max:f64,
    mean_avg_cm:f64,
    std_cm:f64,
    mean_avg_rj:f64,
    std_rj:f64
) -> Result<(), Box<dyn std::error::Error>> {

    //Create the drawing area
    let root = BitMapBackend::new(&filename, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Average <σ_z> trajectory", ("FiraCode Nerd Font", 30))
        .margin(100)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..steps, min..max)?;

    chart.configure_mesh()
        .x_desc("Time steps")
        .y_desc("<σ_z>")
        .label_style(("FiraCode Nerd Font", 30).into_font())
        .draw()?;

    // Draw the average trajectories and their standard deviations following CM algorithm
    chart.draw_series(LineSeries::new(
        avg_cm.iter().enumerate().map(|(x, y)| (x, *y)),
        &BLUE,
    ))?
    .label("CM")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.draw_series(AreaSeries::new(
        (0..avg_cm.len()).map(|i| (i, mean_avg_cm + std_cm)),    // upper curve
        mean_avg_cm - std_cm,        // lower constant baseline
        &BLUE.mix(0.2) // translucent fill
    ))?
    .label("±1σ")
    .legend(|(x, y)| {
        Rectangle::new(
            [(x, y - 5), (x + 20, y + 5)],
            &BLUE.mix(0.2),
        )
    });

    // Draw the average trajectories and their standard deviations following RJ algorithm
    chart.draw_series(LineSeries::new(
        avg_rj.iter().enumerate().map(|(x, y)| (x, *y)),
        &RED,
    ))?
    .label("RJ")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.draw_series(AreaSeries::new(
        (0..avg_rj.len()).map(|i| (i, mean_avg_rj + std_rj)),    // upper curve
        mean_avg_rj - std_rj,        // lower constant baseline
        &RED.mix(0.2) // translucent fill
    ))?
    .label("±1σ")
    .legend(|(x, y)| {
        Rectangle::new(
            [(x, y - 5), (x + 20, y + 5)],
            &RED.mix(0.2),
        )
    });

    // Draw the Lindblad average trajectory
    chart.draw_series(LineSeries::new(
        lindblad_avg.iter().enumerate().map(|(x, y)| (x, *y)),
        &MAGENTA,
    ))?
    .label("Avg")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));

    // Draw the legend
    chart.configure_series_labels()
    .position(SeriesLabelPosition::UpperRight)
    .label_font(("FiraCode Nerd Font", 40).into_font())
    .draw()?;

    Ok(())
}





// No CHUNKS NO wirte files YES prograss bars
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let omega: f64 = 10.0;
    let gamma: f64 = 7.0;
    let dt: f64 = 0.001;
    let total_time: f64 = 30.0;
    let num_trajectories: usize = 10000;
    let m: usize = 5;
    let steps: usize = (total_time / dt).ceil() as usize;

    // Create and configure progress bar
    let pb_cm = ProgressBar::new(num_trajectories as u64);
    pb_cm.set_style(
        ProgressStyle::default_bar()
        .template("Running CM: [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap(),
    );

    let (trajectories_cm, times_jumps_cm): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (0..num_trajectories)
        .into_par_iter()
        .map_init(|| pb_cm.clone(), |pb, _| {
            let result = simulate_spin_jump_cm(omega, gamma, dt, total_time);
            pb.inc(1);
            result
        })
        .unzip();
    pb_cm.finish_with_message("CM simulation complete");

    let pb_rj = ProgressBar::new(num_trajectories as u64);
    pb_rj.set_style(
        ProgressStyle::default_bar()
            .template("Running RJ: [{bar:40.green/black}] {pos}/{len} ({eta})")
            .unwrap(),
    );

    let (trajectories_rj, times_jumps_rj): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (0..num_trajectories)
    .into_par_iter()
        .map_init(|| pb_rj.clone(), |pb, _| {
            let result = simulate_spin_jump_rj(omega, gamma, dt, total_time);
            pb.inc(1);
            result
        })
        .unzip();
    pb_rj.finish_with_message("RJ simulation complete");
    
    let lindblad_avg: Vec<f64> = lindblad_simulation(omega, gamma, dt, total_time);
    
    let pb_ticks_rj = ProgressBar::new(num_trajectories as u64);
    pb_ticks_rj.set_style(
        ProgressStyle::default_bar()
            .template("Analyzing RJ waiting times: [{bar:40.magenta/black}] {pos}/{len} ({eta})")
            .unwrap(),
    );

    let all_waiting_times_rj: Vec<Vec<f64>> = times_jumps_rj
        .par_iter()
        .map_init(|| pb_ticks_rj.clone(), |pb, times| {
            let ticks = compute_tick_times(times, m);
            pb.inc(1);
            analyze_waiting_times(&ticks)
        })
        .collect();
    pb_ticks_rj.finish_with_message("RJ waiting time analysis complete");

    let mut flat_waits_rj: Vec<f64> = all_waiting_times_rj
        .into_par_iter()
        .flatten()
        .collect();
    flat_waits_rj.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let pb_ticks_cm = ProgressBar::new(num_trajectories as u64);
    pb_ticks_cm.set_style(
        ProgressStyle::default_bar()
            .template("Analyzing CM waiting times: [{bar:40.yellow/black}] {pos}/{len} ({eta})")
            .unwrap(),
    );

    let all_waiting_times_cm: Vec<Vec<f64>> = times_jumps_cm
        .par_iter()
        .map_init(|| pb_ticks_cm.clone(), |pb, times| {
            let ticks = compute_tick_times(times, m);
            pb.inc(1);
            analyze_waiting_times(&ticks)
        })
        .collect();
    pb_ticks_cm.finish_with_message("CM waiting time analysis complete");
    
    let mut flat_waits_cm: Vec<f64> = all_waiting_times_cm
        .into_par_iter()
        .flatten()
        .collect();
    
    let mean_wait = flat_waits_cm.iter().sum::<f64>() / flat_waits_cm.len() as f64;

    flat_waits_cm.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let bin_width_cm = bin_width(&flat_waits_cm);
    let bin_width_rj = bin_width(&flat_waits_rj);

    let counts_cm = counts_per_bin(&flat_waits_cm, bin_width_cm, 0.0, total_time);
    let counts_rj = counts_per_bin(&flat_waits_rj, bin_width_rj, 0.0, total_time);

    let filename_cm = format!("histogram_cm_omega-{}_gamma-{}_dt-{}_ntraj-{}.png", omega, gamma, dt, num_trajectories);
    plot_histogram(&counts_cm, bin_width_cm, 0.0, total_time, &filename_cm)?;
    
    let filename_rj = format!("histogram_rj_omega-{}_gamma-{}_dt-{}_ntraj-{}.png", omega, gamma, dt, num_trajectories);
    plot_histogram(&counts_rj, bin_width_rj, 0.0, total_time, &filename_rj)?;

    let flat_cm: Vec<f64> = trajectories_cm.into_iter().flatten().collect();
    let data_cm = Array2::from_shape_vec((num_trajectories, steps), flat_cm)?;
    
    let flat_rj: Vec<f64> = trajectories_rj.into_iter().flatten().collect();
    let data_rj = Array2::from_shape_vec((num_trajectories, steps), flat_rj)?;

    let avg_cm: Array1<f64> = data_cm.mean_axis(Axis(0)).unwrap();
    let avg_rj: Array1<f64> = data_rj.mean_axis(Axis(0)).unwrap();
    
    let mean_avg_cm: f64 = avg_cm.mean().unwrap();
    let mean_avg_rj: f64 = avg_rj.mean().unwrap();

    let std_cm: f64 = avg_cm.var(0.0).sqrt();
    let std_rj: f64 = avg_rj.var(0.0).sqrt();

    let min: f64 = avg_cm.mean().unwrap() - 2.5 * (std_cm + std_rj);
    let max: f64 = avg_cm.mean().unwrap() + 2.5 * (std_cm + std_rj);
    
    println!("CM average trajectory: {}", avg_cm.mean().unwrap());
    println!("CM average trajectory std: {}", avg_cm.var(0.0).sqrt());

    println!("RJ average trajectory: {}", avg_rj.mean().unwrap());
    println!("RJ average trajectory std: {}", avg_rj.var(0.0).sqrt());
    
    let filename = format!("plot_omega-{}_gamma-{}_dt-{}_ntraj-{}.png", omega, gamma, dt, num_trajectories);
    plot_trajectory_avg(avg_cm, avg_rj, lindblad_avg, steps, &filename, min, max, mean_avg_cm, std_cm, mean_avg_rj, std_rj)?;
    
    println!("Simulation completed successfully!");
    println!("Generated files: {}, {}, {}", filename_cm, filename_rj, filename);
    Ok(())
}







// fn clean_directory(dir: &str) -> std::io::Result<()> {
//     if Path::new(dir).exists() {
//         for entry in fs::read_dir(dir)? {
//             let entry = entry?;
//             let path = entry.path();
//             if path.is_file() {
//                 remove_file(path)?;
//             }
//         }
//     }
//     Ok(())
// }


// // With CHUNKS and PROGRESS BARS
// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     // Simulation parameters
//     let omega: f64 = 10.0;
//     let gamma: f64 = 7.0;
//     let dt: f64 = 0.001;
//     let total_time: f64 = 30.0;
//     let num_trajectories: usize = 500;
//     let m: usize = 10;

//     // chunking
//     let chunk_size: usize = 500;
//     let cm_chunks = (num_trajectories + chunk_size - 1) / chunk_size;
//     let rj_chunks = cm_chunks;

//     // precompute steps
//     let steps: usize = (total_time / dt).ceil() as usize;

//     // ─── Prepare directories ──────────────────────────────────────────────────
//     let base_dir = "trajectories";
//     let cm_dir   = format!("{}/cm", base_dir);
//     let rj_dir   = format!("{}/rj", base_dir);

//     // clean & recreate
//     clean_directory(&cm_dir)?;
//     clean_directory(&rj_dir)?;
    
//     create_dir_all(&cm_dir)?;
//     create_dir_all(&rj_dir)?;

//     // ─── Setup progress bars ──────────────────────────────────────────────────
//     let multi_progress = MultiProgress::new();
    
//     let cm_pb = multi_progress.add(ProgressBar::new(cm_chunks as u64));
//     cm_pb.set_style(ProgressStyle::default_bar()
//         .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} CM chunks ({eta})")
//         .unwrap()
//         .progress_chars("#>-"));
//     cm_pb.set_message("CM simulation");

//     let rj_pb = multi_progress.add(ProgressBar::new(rj_chunks as u64));
//     rj_pb.set_style(ProgressStyle::default_bar()
//         .template("{spinner:.green} [{elapsed_precise}] [{bar:40.magenta/blue}] {pos}/{len} RJ chunks ({eta})")
//         .unwrap()
//         .progress_chars("#>-"));
//     rj_pb.set_message("RJ simulation");

//     // ─── CM: simulate & write in chunks ───────────────────────────────────────
//     println!("Starting CM simulation...");
//     (0..cm_chunks).into_par_iter().for_each(|chunk_idx| {
//         let start = chunk_idx * chunk_size;
//         let end   = ((chunk_idx + 1) * chunk_size).min(num_trajectories);
//         let count = end - start;

//         let mut traj_buf  = Vec::with_capacity(count);
//         let mut jumps_buf = Vec::with_capacity(count);
//         for _ in 0..count {
//             let (traj, jtimes) = simulate_spin_jump_cm(omega, gamma, dt, total_time);
//             traj_buf.push(traj);
//             jumps_buf.push(jtimes);
//         }

//         // write trajectories
//         let path_t = format!("{}/trajectory_chunk_{}.csv", cm_dir, chunk_idx);
//         let mut ft = File::create(&path_t).expect("CM traj chunk create failed");
//         writeln!(ft, "traj_id,time,state").unwrap();
//         for (local_id, traj) in traj_buf.iter().enumerate() {
//             let gid = start + local_id;
//             for (i, &state) in traj.iter().enumerate() {
//                 let t = i as f64 * dt;
//                 writeln!(ft, "{},{:.6},{:.6}", gid, t, state).unwrap();
//             }
//         }

//         // write jumps
//         let path_j = format!("{}/jumps_chunk_{}.csv", cm_dir, chunk_idx);
//         let mut fj = File::create(&path_j).expect("CM jumps chunk create failed");
//         writeln!(fj, "traj_id,jump_time").unwrap();
//         for (local_id, times) in jumps_buf.iter().enumerate() {
//             let gid = start + local_id;
//             for &t in times.iter() {
//                 writeln!(fj, "{},{}", gid, t).unwrap();
//             }
//         }

//         // Update progress
//         cm_pb.inc(1);
//     });
//     cm_pb.finish_with_message("CM simulation completed");

//     // ─── RJ: simulate & write in chunks ───────────────────────────────────────
//     println!("Starting RJ simulation...");
//     (0..rj_chunks).into_par_iter().for_each(|chunk_idx| {
//         let start = chunk_idx * chunk_size;
//         let end   = ((chunk_idx + 1) * chunk_size).min(num_trajectories);
//         let count = end - start;

//         let mut traj_buf  = Vec::with_capacity(count);
//         let mut jumps_buf = Vec::with_capacity(count);
//         for _ in 0..count {
//             let (traj, jtimes) = simulate_spin_jump_rj(omega, gamma, dt, total_time);
//             traj_buf.push(traj);
//             jumps_buf.push(jtimes);
//         }

//         // write trajectories
//         let path_t = format!("{}/trajectory_chunk_{}.csv", rj_dir, chunk_idx);
//         let mut ft = File::create(&path_t).expect("RJ traj chunk create failed");
//         writeln!(ft, "traj_id,time,state").unwrap();
//         for (local_id, traj) in traj_buf.iter().enumerate() {
//             let gid = start + local_id;
//             for (i, &state) in traj.iter().enumerate() {
//                 let t = i as f64 * dt;
//                 writeln!(ft, "{},{:.6},{:.6}", gid, t, state).unwrap();
//             }
//         }

//         // write jumps
//         let path_j = format!("{}/jumps_chunk_{}.csv", rj_dir, chunk_idx);
//         let mut fj = File::create(&path_j).expect("RJ jumps chunk create failed");
//         writeln!(fj, "traj_id,jump_time").unwrap();
//         for (local_id, times) in jumps_buf.iter().enumerate() {
//             let gid = start + local_id;
//             for &t in times.iter() {
//                 writeln!(fj, "{},{}", gid, t).unwrap();
//             }
//         }

//         // Update progress
//         rj_pb.inc(1);
//     });
//     rj_pb.finish_with_message("RJ simulation completed");

//     // ─── Reading data back with progress bars ─────────────────────────────────
//     let read_pb = multi_progress.add(ProgressBar::new((cm_chunks + rj_chunks) as u64));
//     read_pb.set_style(ProgressStyle::default_bar()
//         .template("{spinner:.green} [{elapsed_precise}] [{bar:40.yellow/blue}] {pos}/{len} Reading chunks ({eta})")
//         .unwrap()
//         .progress_chars("#>-"));
//     read_pb.set_message("Reading data");

//     // ─── Read back CM data from chunks ────────────────────────────────────────
//     let mut trajectories_cm = vec![Vec::new(); num_trajectories];
//     let mut times_jumps_cm  = vec![Vec::new(); num_trajectories];

//     for chunk_idx in 0..cm_chunks {
//         // trajectories
//         let path = format!("{}/trajectory_chunk_{}.csv", cm_dir, chunk_idx);
//         let mut rdr = ReaderBuilder::new().has_headers(true).from_path(&path)?;
//         for rec in rdr.records() {
//             let row = rec?;
//             let id: usize = row[0].parse()?;
//             let state: f64 = row[2].parse()?;
//             trajectories_cm[id].push(state);
//         }
//         // jumps
//         let path = format!("{}/jumps_chunk_{}.csv", cm_dir, chunk_idx);
//         let mut rdr = ReaderBuilder::new().has_headers(true).from_path(&path)?;
//         for rec in rdr.records() {
//             let row = rec?;
//             let id: usize = row[0].parse()?;
//             let t:  f64   = row[1].parse()?;
//             times_jumps_cm[id].push(t);
//         }
//         read_pb.inc(1);
//     }

//     // ─── Read back RJ data from chunks ────────────────────────────────────────
//     let mut trajectories_rj = vec![Vec::new(); num_trajectories];
//     let mut times_jumps_rj  = vec![Vec::new(); num_trajectories];

//     for chunk_idx in 0..rj_chunks {
//         // trajectories
//         let path = format!("{}/trajectory_chunk_{}.csv", rj_dir, chunk_idx);
//         let mut rdr = ReaderBuilder::new().has_headers(true).from_path(&path)?;
//         for rec in rdr.records() {
//             let row = rec?;
//             let id: usize = row[0].parse()?;
//             let state: f64 = row[2].parse()?;
//             trajectories_rj[id].push(state);
//         }
//         // jumps
//         let path = format!("{}/jumps_chunk_{}.csv", rj_dir, chunk_idx);
//         let mut rdr = ReaderBuilder::new().has_headers(true).from_path(&path)?;
//         for rec in rdr.records() {
//             let row = rec?;
//             let id: usize = row[0].parse()?;
//             let t:  f64   = row[1].parse()?;
//             times_jumps_rj[id].push(t);
//         }
//         read_pb.inc(1);
//     }
//     read_pb.finish_with_message("Data reading completed");

//     // ─── Analysis with progress bar ───────────────────────────────────────────
//     let analysis_pb = multi_progress.add(ProgressBar::new_spinner());
//     analysis_pb.set_style(ProgressStyle::default_spinner()
//         .template("{spinner:.green} [{elapsed_precise}] {msg}")
//         .unwrap());
//     analysis_pb.set_message("Running Lindblad simulation...");

//     // Lindblad average trajectory
//     let lindblad_avg: Vec<f64> = lindblad_simulation(omega, gamma, dt, total_time);
//     analysis_pb.set_message("Computing waiting times...");

//     // Compute waiting times and flatten & sort
//     let mut flat_waits_cm: Vec<f64> = times_jumps_cm
//         .par_iter()
//         .flat_map(|times| analyze_waiting_times(&compute_tick_times(times, m)))
//         .collect();
//     flat_waits_cm.sort_by(|a, b| a.partial_cmp(b).unwrap());

//     let average_cm = flat_waits_cm.iter().copied().sum::<f64>() / flat_waits_cm.len() as f64;
//     let variance_cm = flat_waits_cm.iter().map(|x| (x - average_cm).powi(2)).sum::<f64>() / flat_waits_cm.len() as f64;

//     let accuracy_cm: f64 = average_cm.powi(2) / variance_cm;
//     let resolution_cm: f64 = 1.0 / average_cm;

//     println!("CM Accuracy: {}, Resolution: {}", accuracy_cm, resolution_cm);

//     let mut flat_waits_rj: Vec<f64> = times_jumps_rj
//         .par_iter()
//         .flat_map(|times| analyze_waiting_times(&compute_tick_times(times, m)))
//         .collect();
//     flat_waits_rj.sort_by(|a, b| a.partial_cmp(b).unwrap());

//     let average_rj = flat_waits_rj.iter().copied().sum::<f64>() / flat_waits_rj.len() as f64;
//     let variance_rj = flat_waits_rj.iter().map(|x| (x - average_rj).powi(2)).sum::<f64>() / flat_waits_rj.len() as f64;

//     let accuracy_rj: f64 = average_rj.powi(2) / variance_rj;
//     let resolution_rj: f64 = 1.0 / average_rj;

//     println!("RJ Accuracy: {}, Resolution: {}", accuracy_rj, resolution_rj);

//     analysis_pb.set_message("Creating histograms...");

//     // Histogram binning
//     let bin_width_cm = bin_width(&flat_waits_cm);
//     let bin_width_rj = bin_width(&flat_waits_rj);

//     let counts_cm = counts_per_bin(&flat_waits_cm, bin_width_cm, 0.0, total_time);
//     let counts_rj = counts_per_bin(&flat_waits_rj, bin_width_rj, 0.0, total_time);

//     // Plot histograms
//     let filename_cm = format!(
//         "histogram_cm_omega-{}_gamma-{}_dt-{}_ntraj-{}.png",
//         omega, gamma, dt, num_trajectories
//     );
//     plot_histogram(&counts_cm, bin_width_cm, 0.0, total_time, &filename_cm)?;

//     let filename_rj = format!(
//         "histogram_rj_omega-{}_gamma-{}_dt-{}_ntraj-{}.png",
//         omega, gamma, dt, num_trajectories
//     );
//     plot_histogram(&counts_rj, bin_width_rj, 0.0, total_time, &filename_rj)?;

//     analysis_pb.set_message("Computing averages and plotting...");

//     // Compute average trajectories
//     let flat_cm: Vec<f64> = trajectories_cm.into_iter().flatten().collect();
//     let data_cm = Array2::from_shape_vec((num_trajectories, steps), flat_cm)?;

//     let flat_rj: Vec<f64> = trajectories_rj.into_iter().flatten().collect();
//     let data_rj = Array2::from_shape_vec((num_trajectories, steps), flat_rj)?;

//     let avg_cm: Array1<f64> = data_cm.mean_axis(Axis(0)).unwrap();
//     let avg_rj: Array1<f64> = data_rj.mean_axis(Axis(0)).unwrap();

//     // Plot average trajectories
//     let filename = format!(
//         "plot_omega-{}_gamma-{}_dt-{}_ntraj-{}.png",
//         omega, gamma, dt, num_trajectories
//     );
//     plot_trajectory_avg(avg_cm, avg_rj, lindblad_avg, steps, &filename)?;

//     analysis_pb.finish_with_message("Analysis completed");

//     println!("Simulation completed successfully!");
//     println!("Generated files: {}, {}, {}", filename_cm, filename_rj, filename);
//     Ok(())
// }




// // No CHUNKS NO wirte files NO prograss bars
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
