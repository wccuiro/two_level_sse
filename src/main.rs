use ndarray::{array, Array1, Array2, Axis};
use ndarray::linalg::kron;

use num_complex::Complex64;
use rand::Rng;

use rayon::prelude::*;
use plotters::prelude::*;

fn simulate_spin_jump_cm(
    omega: f64,
    gamma: f64,
    dt: f64,
    total_time: f64,
) -> Vec<f64> {
    let sigma_x: Array2<Complex64> = array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ];
    let sigma_z: Array2<Complex64> = array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
    ];
    let sigma_plus: Array2<Complex64> = array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
    ];
    let sigma_minus: Array2<Complex64> = array![
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ];

    let steps = (total_time / dt).ceil() as usize;

    let mut psi: Array1<Complex64>  = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let mut sz_exp = Vec::with_capacity(steps);

    let sigma_pm = sigma_plus.dot(&sigma_minus);
    let h_eff = sigma_x.mapv(|e| e * omega) - sigma_pm.mapv(|e| e * (gamma * Complex64::new(0.0, 0.5)));

    let mut rng = rand::thread_rng();

    for _ in 0..steps {
        let amp = psi.mapv(|x| x.conj()).dot(&sigma_pm.dot(&psi)).re;
        let p_jump = gamma * amp * dt;
        let dpsi_nh = (&psi.mapv(|e| e * (gamma * amp * 0.5))
            - &(h_eff.dot(&psi).mapv(|e| Complex64::new(0.0, 1.0) * e)))
            .mapv(|e| e * dt);

        if rng.gen::<f64>() < p_jump {
            let norm_factor = (gamma * amp).sqrt();
            psi = sigma_minus.dot(&psi).mapv(|e| e * (gamma.sqrt() / norm_factor));
        }

        psi = &psi + &dpsi_nh;
        let norm = psi.mapv(|x| x.conj()).dot(&psi).re.sqrt();
        psi = psi.mapv(|e| e / norm);

        let szz = psi.mapv(|x| x.conj()).dot(&sigma_z.dot(&psi)).re;
        sz_exp.push(szz);
    }

    sz_exp
}

fn simulate_spin_jump_rj(
    omega: f64,
    gamma: f64,
    dt: f64,
    total_time: f64,
) -> Vec<f64> {
    let max_steps = (total_time / dt).ceil() as usize;
    let mut sz_exp = Vec::with_capacity(max_steps);

    let sigma_x = array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ];
    let sigma_z = array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
    ];
    let sigma_plus = array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
    ];
    let sigma_minus = array![
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ];

    let sigma_pm = sigma_plus.dot(&sigma_minus);
    let h_eff = sigma_x.mapv(|e| e * omega) - sigma_pm.mapv(|e| e * (gamma * Complex64::new(0.0, 0.5)));

    let mut psi = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let mut rng = rand::thread_rng();

    let mut p0 = 1.0;
    let mut r = rng.gen::<f64>();
    let mut nh_evol = true;

    for _ in 0..max_steps {
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

    sz_exp
}


fn lindblad_simulation(
    omega: f64,
    gamma: f64,
    dt: f64,
    total_time: f64,
) -> Vec<f64> {
    let max_steps = (total_time / dt).ceil() as usize;
    let mut sz_exp = Vec::with_capacity(max_steps);

    let sigma_x = array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ];
    let sigma_z = array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
    ];
    let sigma_plus = array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
    ];
    let sigma_minus = array![
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ];

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

        let szz = v_identity.dot(&v_sigmaz.dot(&rho)).re;
        sz_exp.push(szz);

    }

    sz_exp
}






fn main() -> Result<(), Box<dyn std::error::Error>> {
    let omega: f64 = 0.7;
    let gamma: f64 = 0.2;
    let dt: f64 = 0.0001;
    let total_time: f64 = 30.0;
    let num_trajectories: usize = 2000;

    // Calculate number of steps as usize
    let steps: usize = (total_time / dt).ceil() as usize;

    // Run simulations in parallel, passing total_time
    let trajectories_cm: Vec<Vec<f64>> = (0..num_trajectories)
        .into_par_iter()
        .map(|_| simulate_spin_jump_cm(omega, gamma, dt, total_time))
        .collect();

    // Run simulations in parallel, passing total_time
    let trajectories_rj: Vec<Vec<f64>> = (0..num_trajectories)
        .into_par_iter()
        .map(|_| simulate_spin_jump_rj(omega, gamma, dt, total_time))
        .collect();
    
    let lindblad_avg: Vec<f64> = lindblad_simulation(omega, gamma, dt, total_time);

    // // Run simulations in parallel, passing total_time
    // let trajectories_rjop: Vec<Vec<f64>> = (0..num_trajectories)
    //     .into_par_iter()
    //     .map(|_| simulate_spin_jump_optimized(omega, gamma, dt, total_time))
    //     .collect();

    // Flatten and reshape into 2D array
    let flat_cm: Vec<f64> = trajectories_cm.into_iter().flatten().collect();
    let data_cm = Array2::from_shape_vec((num_trajectories, steps), flat_cm)?;

    let flat_rj: Vec<f64> = trajectories_rj.into_iter().flatten().collect();
    let data_rj = Array2::from_shape_vec((num_trajectories, steps), flat_rj)?;

    // let flat_rjop: Vec<f64> = trajectories_rjop.into_iter().flatten().collect();
    // let data_rjop = Array2::from_shape_vec((num_trajectories, steps), flat_rjop)?;

    // Mean over rows (trajectories)
    let avg_cm: Array1<f64> = data_cm.mean_axis(Axis(0)).unwrap();

    let avg_rj: Array1<f64> = data_rj.mean_axis(Axis(0)).unwrap();

    // let avg_rjop: Array1<f64> = data_rjop.mean_axis(Axis(0)).unwrap();

    // Plot the average trajectory
    let filename = format!("plot_dt-{:.5}_ntraj-{}.png", dt, num_trajectories);
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

    println!("SSE.png");
    Ok(())
}
