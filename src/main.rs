use ndarray::{array, Array1, Array2, Axis};
use ndarray::linalg::kron;

use num_complex::Complex64;
use rand::Rng;

use rayon::prelude::*;
use plotters::prelude::*;

use indicatif::{ProgressBar, ProgressStyle};


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

fn simulate_spin_jump_rj(
    omega: f64,
    gamma: f64,
    dt: f64,
    total_time: f64,
) -> (Vec<f64>, Array1<f64>, Vec<Array1<Complex64>>) {
    let max_steps = (total_time / dt).ceil() as usize;

    let mut sz_exp = Vec::with_capacity(max_steps);

    let mut time_jumps:Vec<f64> = Vec::new();
    let mut wfs = Vec::new();

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

    for i in 0..max_steps {

        let amp = psi.mapv(|x| x.conj()).dot(&sigma_pm.dot(&psi)).re;

        let prob = gamma * amp * dt;

        let dpsi_nh = (&psi.mapv(|e| e * (gamma * amp * 0.5))
            - &(h_eff.dot(&psi).mapv(|e| Complex64::new(0.0, 1.0) * e)))
            .mapv(|e| e * dt);

        if r >= p0 {
            let norm_factor = amp.sqrt();
            psi = sigma_minus.dot(&psi).mapv(|e| e * (1. / norm_factor));
            psi = &psi + &dpsi_nh;

            psi /= psi.mapv(|x| x.conj()).dot(&psi).sqrt();

            r = rng.gen::<f64>();
            
            p0 = 1.0;

            time_jumps.push(i as f64 * dt);
            wfs.push(psi.clone()); 
        } else {
            psi = &psi + &dpsi_nh;
            psi /= psi.mapv(|x| x.conj()).dot(&psi).sqrt();
        }

        p0 *= 1.0 - prob;

        let szz = psi.mapv(|x| x.conj()).dot(&sigma_z.dot(&psi)).re;
        sz_exp.push(szz);
    }

    let time_jumps: Array1<f64> = Array1::from(time_jumps);

    (sz_exp, time_jumps, wfs)
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

    let total: f64 = counts.iter().sum::<usize>() as f64 * bin_width;
    let norm_counts: Vec<f64> = counts.iter().map(|&e| e as f64 / total).collect();

    norm_counts
}

fn compute_tick_times(
    times: &Array1<f64>,
    m: usize,
) -> Array1<usize> {
    let mut n_acc: usize = 0;
    let mut aux_ticks = vec![];
    let mut next_threshold = m;

    for i in 0.. times.len() {
        n_acc += 1;

        if n_acc >= next_threshold {
            aux_ticks.push(i);
            next_threshold += m;
        }
    }

    let aux_ticks: Array1<usize> = Array1::from(aux_ticks);
    aux_ticks
}

fn analyze_ticks(
    times: &Array1<f64>,
    wfs: &Vec<Array1<Complex64>>,
    aux_ticks: &Array1<usize>,
    pi: &Array2<Complex64>,
) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
    let mut waiting_times = Vec::new();
    let mut activity_ticks = Vec::new();
    let mut entropy_ticks = Vec::new();

    let ticks = aux_ticks.as_slice().expect("aux_ticks must be contiguous");

    for pair in ticks.windows(2) {
        if pair.len() != 2 {
            continue; // Skip incomplete pair
        }
        let i1 = pair[0];
        let i2 = pair[1];

        let t1 = times[i1];
        let t2 = times[i2];

        let psi1 = &wfs[i1];
        let psi2 = &wfs[i2];

        let p1 = {
            let inner = pi.dot(psi1);
            let amp = psi1.mapv(|c| c.conj()).dot(&inner).re;
            amp.clamp(1e-12, 1.0)
        };
        let p2 = {
            let inner = pi.dot(psi2);
            let amp = psi2.mapv(|c| c.conj()).dot(&inner).re;
            amp.clamp(1e-12, 1.0)
        };

        let delta_spsi = p1.ln() - p2.ln();
        let s_tick = delta_spsi;

        waiting_times.push(t2 - t1);
        activity_ticks.push(i2 - i1);
        entropy_ticks.push(s_tick);
    }
    

    (waiting_times, activity_ticks, entropy_ticks)
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
    avg_rj: Array1<f64>,
    lindblad_avg: Vec<f64>,
    steps: usize,
    filename: &str,
    min:f64,
    max:f64,
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

    // // Draw the average trajectories and their standard deviations following CM algorithm
    // chart.draw_series(LineSeries::new(
    //     avg_cm.iter().enumerate().map(|(x, y)| (x, *y)),
    //     &BLUE,
    // ))?
    // .label("CM")
    // .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // chart.draw_series(AreaSeries::new(
    //     (0..avg_cm.len()).map(|i| (i, mean_avg_cm + std_cm)),    // upper curve
    //     mean_avg_cm - std_cm,        // lower constant baseline
    //     &BLUE.mix(0.2) // translucent fill
    // ))?
    // .label("±1σ")
    // .legend(|(x, y)| {
    //     Rectangle::new(
    //         [(x, y - 5), (x + 20, y + 5)],
    //         &BLUE.mix(0.2),
    //     )
    // });

    // Draw the average trajectories and their standard deviations following RJ algorithm
    chart.draw_series(LineSeries::new(
        avg_rj.iter().enumerate().map(|(x, y)| (x, *y)),
        &BLUE,
    ))?
    .label("RJ")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.draw_series(AreaSeries::new(
        (0..avg_rj.len()).map(|i| (i, mean_avg_rj + std_rj)),    // upper curve
        mean_avg_rj - std_rj,        // lower constant baseline
        &BLUE.mix(0.2) // translucent fill
    ))?
    .label("±1σ")
    .legend(|(x, y)| {
        Rectangle::new(
            [(x, y - 5), (x + 20, y + 5)],
            &BLUE.mix(0.2),
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

fn plot_entropy_vs_n_traj(
    entropies_traj: Vec<f64>,
    n_traj: Vec<usize>,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(entropies_traj.len(), n_traj.len());

    // Set up the plot
    let root = BitMapBackend::new(filename, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_range = *n_traj.iter().min().unwrap() as f64..*n_traj.iter().max().unwrap() as f64;
    let y_range = entropies_traj
        .iter()
        .cloned()
        .fold(f64::INFINITY..f64::NEG_INFINITY, |acc, v| {
            acc.start.min(v)..acc.end.max(v)
        });

    let mut chart = ChartBuilder::on(&root)
        .caption("Transformed Entropy vs. Number of Trajectories", ("sans-serif", 30))
        .margin(40)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_range.clone(), y_range.clone())?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        n_traj.iter().zip(entropies_traj.iter()).map(|(x, y)| (*x as f64, *y)),
        &RED,
    ))?;

    Ok(())
}

fn plot_histogram_omega_gamma(
    counts_val: &[Vec<f64>],
    bin_width_val: &[f64],
    total_time: f64,
    filename_rj: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        counts_val.len(),
        bin_width_val.len(),
        "counts_val and bin_width_val must have the same length"
    );

    // Compute global max_count across all histograms
    let global_max = counts_val
        .iter()
        .flat_map(|counts| counts.iter())
        .cloned()
        .fold(0.0_f64, f64::max);

    // Single drawing area
    let root = BitMapBackend::new(filename_rj, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    // Build one chart spanning the full area
    let mut chart = ChartBuilder::on(&root)
        .caption("Changing both", ("sans-serif", 40))
        .margin(30)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..total_time, 0.0..global_max)?;

    chart
        .configure_mesh()
        .x_desc("Time")
        .y_desc("Count")
        .draw()?;

    // A palette of base colors (RGBColor refs)
    let palette = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN];

    for (idx, (counts, &bin_width)) in
        counts_val.iter().zip(bin_width_val.iter()).enumerate()
    {
        // pick the base color for this series
        let color = palette[idx % palette.len()];

        // build a semi‑transparent ShapeStyle once
        let bar_style = color.mix(0.6).filled();

        // draw each bar with that style
        for (i, &count) in counts.iter().enumerate() {
            let x0 = i as f64 * bin_width;
            let x1 = x0 + bin_width;
            chart.draw_series(std::iter::once(
                Rectangle::new([(x0, 0.0), (x1, count)], bar_style.clone()),
            ))?;
        }

        // add a legend entry using the *same* style
        chart
            .draw_series(std::iter::once(Circle::new(
                (total_time * 0.05, global_max * (0.95 - 0.05 * idx as f64)),
                5,
                bar_style.clone(),
            )))?
            .label(format!("alpha {}", idx))
            .legend(move |(x, y)| {
                Rectangle::new([(x, y - 5), (x + 10, y + 5)], bar_style.clone())
            });
    }

    // then draw the legend box once at the end
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}


// No CHUNKS NO wirte files YES prograss bars
fn main() -> Result<(), Box<dyn std::error::Error>> {
    //This variables are fixed for all siulations
    let dt: f64 = 0.001;
    let total_time: f64 = 30.0;
    let steps: usize = (total_time / dt).ceil() as usize;

    
    // Number of simulations
    let n_pts = 20_usize;

    // // Reference values for the parameters
    // let omega: f64 = 7.0;
    // let gamma: f64 = 1.0;
    // let num_trajectories: usize = 1000;
    // let m: usize = 5;

    let init_omega  = 5.0_f64;
    let last_omega  = 5.0_f64;

    let vec_omega: Vec<f64> = (0..n_pts)
        .map(|i| {
            let t = i as f64 / (n_pts - 1) as f64;
            init_omega + t * (last_omega - init_omega)
        })
        .collect();
    
    let init_gamma  = 5.0_f64;
    let last_gamma  = 5.0_f64;

    let vec_gamma: Vec<f64> = (0..n_pts)
        .map(|i| {
            let t = i as f64 / (n_pts - 1) as f64;
            init_gamma + t * (last_gamma - init_gamma)
        })
        .collect();

    let init_num_trajectories  = 100_usize;
    let last_num_trajectories  = 1000_usize;

    let vec_num_trajectories: Vec<usize> = (0..n_pts)
        .map(|i| {
            let t = i as f64 / (n_pts - 1) as f64;
            (init_num_trajectories as f64 + t * (last_num_trajectories - init_num_trajectories) as f64) as usize
        })
        .collect();

    let init_m  = 5_usize;
    let last_m  = 5_usize;

    let vec_m: Vec<usize> = (0..n_pts)
        .map(|i| {
            let t = i as f64 / (n_pts - 1) as f64;
            (init_m as f64 + t * (last_m - init_m) as f64) as usize
        })
        .collect();

    println!("Running simulations with omega: {:?} and gamma: {:?}", vec_omega, vec_gamma);

    let mut counts_val = Vec::with_capacity(n_pts);
    let mut bin_width_val = Vec::with_capacity(n_pts);

    let mut avg_jumps_traj = Vec::with_capacity(n_pts);
    let mut entropys_traj = Vec::with_capacity(n_pts);

    let mut accuracy_traj = Vec::with_capacity(n_pts);
    let mut resolution_traj = Vec::with_capacity(n_pts);

    
    for (((&gamma, &omega), &num_trajectories), &m) in vec_gamma.iter()
                            .zip(vec_omega.iter())
                            .zip(vec_num_trajectories.iter())
                            .zip(vec_m.iter()) {


        let (pi, _, _, _) = steady_state(gamma, omega);

        let pb_rj = ProgressBar::new(num_trajectories as u64);
        let tpl = format!(
            "Running RJ {}, {}: [{{bar:40.green/black}}] {{pos}}/{{len}} ({{eta}})",
            gamma, omega
        );
        pb_rj.set_style(
            ProgressStyle::default_bar()
                .template(&tpl)
                .unwrap(),
        );

        // 1) run your parallel sim and collect all the (A,B,C) tuples
        let results: Vec<(Vec<f64>, Array1<f64>, Vec<Array1<Complex64>>)> = 
            (0..num_trajectories)
                .into_par_iter()
                .map_init(|| pb_rj.clone(), |pb, _| {
                    let res = simulate_spin_jump_rj(omega, gamma, dt, total_time);
                    pb.inc(1);
                    res
                })
                .collect();

        // 2) allocate output Vecs up‑front for efficiency
        let mut trajectories_rj  = Vec::with_capacity(num_trajectories);
        let mut times_jumps_rj   = Vec::with_capacity(num_trajectories);
        let mut psi_states_rj    = Vec::with_capacity(num_trajectories);

        let mut len_jumps = Vec::new();

        // 3) unzip by hand
        for (traj, times, psi) in results {
            trajectories_rj.push(traj);
            len_jumps.push(times.len());
            times_jumps_rj.push(times);
            psi_states_rj.push(psi);
        }
        pb_rj.finish_with_message("RJ simulation complete");
        
        avg_jumps_traj.push(len_jumps.iter().sum::<usize>() as f64 / num_trajectories as f64);
        // println!("Average number of jumps per trajectory: {}", len_jumps.iter().sum::<usize>() as f64 / num_trajectories as f64);

        let lindblad_avg: Vec<f64> = lindblad_simulation(omega, gamma, dt, total_time);
        
        let pb_ticks_rj = ProgressBar::new(num_trajectories as u64);
        pb_ticks_rj.set_style(
            ProgressStyle::default_bar()
                .template("Analyzing RJ waiting times: [{bar:40.magenta/black}] {pos}/{len} ({eta})")
                .unwrap(),
        );

        let results: Vec<(Vec<f64>, Vec<usize>, Vec<f64>)> =
            times_jumps_rj
                .into_par_iter()                        // take ownership of each Vec<f64>
                .zip(psi_states_rj.into_par_iter())     // pair it with each Vec<Complex64>
                .map_init(
                    || pb_ticks_rj.clone(),             // make a new PB handle per thread
                    |pb, (times, wfs)| {                // times: Vec<f64>, wfs: Vec<Array1<...>>
                        let aux = compute_tick_times(&times, m);
                        pb.inc(1);
                        analyze_ticks(&times, &wfs, &aux, &pi)
                    },
                )
                .collect();
        pb_ticks_rj.finish_with_message("RJ waiting time analysis complete");

        // Combine all results
        let mut flat_waits_rj = Vec::new();        // flattened for histogram
        let mut _activities = Vec::new(); // one avg per trajectory
        let mut entropies = Vec::new(); // one avg per trajectory

        for (wts, acts, ents) in results {
            flat_waits_rj.extend(wts); // flatten all waits

            // Average number of actions per trajectory
            if !acts.is_empty() {
                let avg_act = acts.iter().map(|x| *x as f64).sum::<f64>();
                _activities.push(avg_act);
            }

            entropies.extend(ents); // flatten all entropies
        }
        
        let mean_wait = flat_waits_rj.iter().sum::<f64>() / flat_waits_rj.len() as f64;
        let var_wait = flat_waits_rj
                                    .iter()
                                    .map(|&x| (x - mean_wait).powi(2))
                                    .sum::<f64>()
                                    / ((flat_waits_rj.len() - 1) as f64);

        let arr_ent = Array1::from(entropies); // assuming entropies: Vec<f64>

        let sum_exp_m_ent = (arr_ent.mapv(|e| (-e).exp()).sum().ln() - 
            (arr_ent.len() as f64).ln()).exp() ;// Mean of entropies, using log mean

        entropys_traj.push(sum_exp_m_ent);
    

        flat_waits_rj.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let accuracy: f64 = mean_wait.powi(2) / var_wait;
        let resolution: f64 = 1.0/mean_wait;

        accuracy_traj.push(mean_wait);
        resolution_traj.push(resolution);

        let max_val: f64 = flat_waits_rj[flat_waits_rj.len() - 1];

        let bin_width_rj = bin_width(&flat_waits_rj);

        let counts_rj = counts_per_bin(&flat_waits_rj, bin_width_rj, 0.0, total_time);

        counts_val.push(counts_rj.clone());
        bin_width_val.push(bin_width_rj.clone());
        
        let filename_rj = format!("histogram_rj_omega-{}_gamma-{}_dt-{}_ntraj-{}.png", omega, gamma, dt, num_trajectories);
        // plot_histogram(&counts_rj, bin_width_rj, 0.0, max_val, &filename_rj)?;

        let flat_rj: Vec<f64> = trajectories_rj.into_iter().flatten().collect();
        let data_rj = Array2::from_shape_vec((num_trajectories, steps), flat_rj)?;

        let avg_rj: Array1<f64> = data_rj.mean_axis(Axis(0)).unwrap();
        
        let mean_avg_rj: f64 = avg_rj.mean().unwrap();

        let std_rj: f64 = avg_rj.var(0.0).sqrt();

        let min: f64 = avg_rj.mean().unwrap() - 2.5 * std_rj;
        let max: f64 = avg_rj.mean().unwrap() + 2.5 * std_rj;
                
        let filename = format!("plot_omega-{}_gamma-{}_dt-{}_ntraj-{}.png", omega, gamma, dt, num_trajectories);
        // plot_trajectory_avg(avg_rj, lindblad_avg, steps, &filename, min, max, mean_avg_rj, std_rj)?;
    }
    
    println!("{}", counts_val.len());
    println!("{:?}", avg_jumps_traj);
    println!("{:?}", entropys_traj);
    println!("{:?}", accuracy_traj);
    println!("{:?}", resolution_traj);

    plot_entropy_vs_n_traj(entropys_traj, vec_num_trajectories, "entropy_vs_n_traj.png")?;

    plot_histogram_omega_gamma(&counts_val, &bin_width_val, total_time, "Changing both.png")?;
    
    println!("Simulation completed successfully!");
    
    Ok(())
}
