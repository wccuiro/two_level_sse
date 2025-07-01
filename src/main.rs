use ndarray::{array, Array1, Array2, Axis};
use ndarray::linalg::kron;

use num_complex::Complex64;
use rand::Rng;

use rayon::prelude::*;
use plotters::prelude::*;


fn gen_sigma_x() -> Array2<Complex64> {
    array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ]
}

fn gen_sigma_z() -> Array2<Complex64> {
    array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
    ]
}

fn gen_sigma_y() -> Array2<Complex64> {
    array![
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
        [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)]
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

    let sigma_plus = gen_sigma_x().mapv(|e| e * Complex64::new(0.5, 0.0)) + gen_sigma_y().mapv(|e| e * Complex64::new(0.5, 1.0)) ;

    let sigma_minus = gen_sigma_x().mapv(|e| e * Complex64::new(0.5, 0.0)) - gen_sigma_y().mapv(|e| e * Complex64::new(0.5, 1.0)) ;

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

        if rng.gen::<f64>() < p_jump {
            let norm_factor = (gamma * amp).sqrt();
            psi = sigma_minus.dot(&psi).mapv(|e| e * (gamma.sqrt() / norm_factor));
            time_jumps.push(i as f64 * dt); 
        }

        psi = &psi + &dpsi_nh;
        let norm = psi.mapv(|x| x.conj()).dot(&psi).re.sqrt();
        psi = psi.mapv(|e| e / norm);

        let szz = psi.mapv(|x| x.conj()).dot(&sigma_z.dot(&psi)).re;
        sz_exp.push(szz);
    }

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

    let sigma_plus = gen_sigma_x().mapv(|e| e * Complex64::new(0.5, 0.0)) + gen_sigma_y().mapv(|e| e * Complex64::new(0.5, 1.0)) ;

    let sigma_minus = gen_sigma_x().mapv(|e| e * Complex64::new(0.5, 0.0)) - gen_sigma_y().mapv(|e| e * Complex64::new(0.5, 1.0)) ;

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
            let norm_factor = (gamma * amp).sqrt();
            psi = sigma_minus.dot(&psi).mapv(|e| e * (gamma.sqrt() / norm_factor));
            time_jumps.push(i as f64 * dt); 
            nh_evol = true;
        }

        psi = &psi + &dpsi_nh;
        psi = psi.mapv(|e| {
            let nn = psi.mapv(|x| x.norm_sqr()).sum().sqrt();
            e / nn
        });

        let szz = psi.mapv(|x| x.conj()).dot(&sigma_z.dot(&psi)).re;
        sz_exp.push(szz);

        let p_jump = gamma * amp * dt;
        p0 *= 1.0 - p_jump;
    }

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

    let sigma_plus = gen_sigma_x().mapv(|e| e * Complex64::new(0.5, 0.0)) + gen_sigma_y().mapv(|e| e * Complex64::new(0.5, 1.0)) ;

    let sigma_minus = gen_sigma_x().mapv(|e| e * Complex64::new(0.5, 0.0)) - gen_sigma_y().mapv(|e| e * Complex64::new(0.5, 1.0)) ;

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

    let term1 = kron(&h_eff, &identity) - kron(&identity, &h_eff.t());
    let term2 = kron(&sigma_minus, &sigma_plus.t());

    let left = sigma_plus.dot(&sigma_minus);
    let right = sigma_minus.t().dot(&sigma_plus.t());

    let term3 = (kron(&left, &identity) + kron(&identity, &right)).mapv(|e| e * 0.5);

    let s_l = &term1.mapv(|e| e * Complex64::new(0.0, -1.0)) + (&term2 - &term3).mapv(|e| e * gamma);

    let mut rho = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];

    for _ in 0..max_steps {
        rho = &rho + (&s_l.dot(&rho)).mapv(|e| e * dt);
        rho = rho.mapv(|e| e / v_identity.dot(&rho).re);

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

    let q1 = quantile(data, 0.25);
    let q3 = quantile(data, 0.75);
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


fn main() -> Result<(), Box<dyn std::error::Error>>{
    let omega: f64 = 0.7;
    let gamma: f64 = 0.2;
    let dt: f64 = 0.01;
    let total_time: f64 = 30.0;
    let num_trajectories: usize = 300;

    // Calculate number of steps as usize
    let steps: usize = (total_time / dt).ceil() as usize;

    // Run simulations in parallel, passing total_time
    let (trajectories_cm, times_jumps_cm): (Vec<Vec<f64>> , Vec<Vec<f64>>) = (0..num_trajectories)
        .into_par_iter()
        .map(|_| simulate_spin_jump_cm(omega, gamma, dt, total_time))
        .unzip();

    // Run simulations in parallel, passing total_time
    let (trajectories_rj, times_jumps_rj): (Vec<Vec<f64>> , Vec<Vec<f64>>) = (0..num_trajectories)
        .into_par_iter()
        .map(|_| simulate_spin_jump_rj(omega, gamma, dt, total_time))
        .unzip();
    
    let lindblad_avg: Vec<f64> = lindblad_simulation(omega, gamma, dt, total_time);

    let mut flat_times_jumps_cm: Vec<f64> = times_jumps_cm.into_iter().flatten().collect();
    flat_times_jumps_cm.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut flat_times_jumps_rj: Vec<f64> = times_jumps_rj.into_iter().flatten().collect();
    flat_times_jumps_rj.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let bin_width_cm = bin_width(&flat_times_jumps_cm);
    let bin_width_rj = bin_width(&flat_times_jumps_rj);


    let counts_cm = counts_per_bin(
        &flat_times_jumps_cm,
        bin_width_cm,
        0.0,
        total_time,
    );

    let counts_rj = counts_per_bin(
        &flat_times_jumps_rj,
        bin_width_rj,
        0.0,
        total_time,
    );

    let filename_cm = format!("histogram_cm_omega-{}_gamma-{}_dt-{}_ntraj-{}.png", omega, gamma, dt, num_trajectories);
    plot_histogram(&counts_cm, bin_width_cm, 0.0, total_time, &filename_cm)?;

    let filename_rj = format!("histogram_rj_omega-{}_gamma-{}_dt-{}_ntraj-{}.png", omega, gamma, dt, num_trajectories);
    plot_histogram(&counts_rj, bin_width_rj, 0.0, total_time, &filename_rj)?;


    // Flatten and reshape into 2D array
    let flat_cm: Vec<f64> = trajectories_cm.into_iter().flatten().collect();
    let data_cm = Array2::from_shape_vec((num_trajectories, steps), flat_cm)?;

    let flat_rj: Vec<f64> = trajectories_rj.into_iter().flatten().collect();
    let data_rj = Array2::from_shape_vec((num_trajectories, steps), flat_rj)?;

    // Mean over rows (trajectories)
    let avg_cm: Array1<f64> = data_cm.mean_axis(Axis(0)).unwrap();

    let avg_rj: Array1<f64> = data_rj.mean_axis(Axis(0)).unwrap();

    // Plot the average trajectory
    let filename = format!("plot_omega-{}_gamma-{}_dt-{}_ntraj-{}.png", omega, gamma, dt, num_trajectories);
    plot_trajectory_avg(avg_cm, avg_rj, lindblad_avg, steps, &filename)?;

    println!("Simulation completed successfully!");
    println!("Generated files: {}, {}, {}", filename_cm, filename_rj, filename);
    Ok(())
}
