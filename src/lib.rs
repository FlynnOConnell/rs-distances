use ndarray::{Array3};
use pyo3::prelude::*;
use numpy::{PyArray3, IntoPyArray};


fn iterate_spiketrains(scr: &mut Array3<f64>, sd: &Array3<f64>)  {
    let (num_qvals, num_spikes_xii, num_spikes_xjj) = scr.dim();
    for xii in 1..num_spikes_xii {
        for xjj in 1..num_spikes_xjj {
            for q in 0..num_qvals {
                let a = scr[[q, xii - 1, xjj]] + 1.0;
                let b = scr[[q, xii, xjj - 1]] + 1.0;
                let c = scr[[q, xii - 1, xjj - 1]] + sd[[q, xii - 1, xjj - 1]];

                scr[[q, xii, xjj]] = a.min(b.min(c));
            }
        }
    }
}

#[pyfunction]
fn iterate_spiketrains_impl(py: Python, scr: &PyArray3<f64>, sd: &PyArray3<f64>) -> PyResult<PyObject> {
    let mut scr: Array3<f64> = scr.to_owned_array();
    let sd: Array3<f64> = sd.to_owned_array();

    iterate_spiketrains(&mut scr, &sd);

    // Convert the result to a PyArray
    let res = scr.into_pyarray(py).to_owned();
    Ok(res.into())
}


#[pymodule]
fn rs_distances(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(iterate_spiketrains_impl, m)?)?;
    Ok(())
}
