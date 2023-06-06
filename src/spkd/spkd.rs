use ndarray::Array2;
use ndarray::{s, Array1, Array3, ArrayBase, ArrayView3, Dim, OwnedRepr};
use numpy::{IntoPyArray, PyArray1, PyArray3};
use pyo3::{prelude::*, types::PyList};

pub fn iterate_spiketrains(scr: &mut Array3<f64>, sd: &ArrayView3<f64>) {
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
fn iterate_spiketrains_impl(
    py: Python,
    scr: &PyArray3<f64>,
    sd: &PyArray3<f64>,
) -> PyResult<PyObject> {
    let mut scr: Array3<f64> = scr.to_owned_array();
    let sd: Array3<f64> = sd.to_owned_array();

    iterate_spiketrains(&mut scr, &sd.view());

    // Convert the result to a PyArray
    let res: Py<numpy::PyArray<f64, Dim<[usize; 3]>>> = scr.into_pyarray(py).to_owned();
    Ok(res.into())
}

pub fn calculate_spkd(
    cspks: &Vec<Array1<f64>>,
    qvals: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,
    num_vectors: usize,
    _res: Option<f64>,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>> {
    let numt: usize = num_vectors; // number of spike trains
    let num_qvals: usize = qvals.len(); // number of q values

    let mut d: ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>> =
        Array3::<f64>::zeros((numt, numt, num_qvals));

    for xi in 0..numt - 1 {
        for xj in xi + 1..numt {
            let curcounts_xi: usize = cspks[xi].len();
            let curcounts_xj: usize = cspks[xj].len();

            if curcounts_xi != 0 && curcounts_xj != 0 {
                let mut outer_diff: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
                    cspks[xi].clone().into_shape((curcounts_xi, 1)).unwrap()
                        - cspks[xj].clone().into_shape((1, curcounts_xj)).unwrap();

                outer_diff.mapv_inplace(|x| x.abs());

                let sd = qvals.clone().into_shape((num_qvals, 1, 1)).unwrap() * &outer_diff;

                let mut scr = Array3::<f64>::zeros((num_qvals, curcounts_xi + 1, curcounts_xj + 1));

                scr.slice_mut(s![.., 1.., 0])
                    .assign(&Array2::from_elem((num_qvals, curcounts_xi), 1.0));
                scr.slice_mut(s![.., 0, 1..])
                    .assign(&Array2::from_elem((num_qvals, curcounts_xj), 1.0));

                iterate_spiketrains(&mut scr, &sd.view());

                let final_values: Array1<f64> = Array1::from_shape_vec(
                    num_qvals,
                    (0..num_qvals)
                        .map(|q| scr[[q, scr.shape()[1] - 1, scr.shape()[2] - 1]])
                        .collect(),
                )
                .unwrap();

                d.slice_mut(s![xi, xj, ..]).assign(&final_values);
            } else {
                println!("d's shape: {:?}", d.dim());
                d.slice_mut(s![xi, xj, ..])
                    .fill(curcounts_xi.max(curcounts_xj) as f64);
            }
        }
    }

    // Transpose d
    d = d.permuted_axes([1, 0, 2]);
    d.mapv_inplace(|x| x.max(0.0));
    d
}

#[pyfunction]
fn calculate_spkd_impl(
    py: Python,
    cspks: &PyList,
    qvals: &PyArray1<f64>,
    res: Option<f64>,
) -> PyResult<PyObject> {
    let mut cspk_vectors: Vec<Array1<f64>> = Vec::new();
    let mut num_vectors: usize = 0;

    // Iterate over the spike trains and convert them to ndarrays
    for pyarray in cspks.iter() {
        let numpy_array: &PyArray1<f64> = pyarray.extract()?;
        let array: Array1<f64> = numpy_array.to_owned_array();
        let shape = array.shape();

        if shape.is_empty() || shape[0] == 0 {
            continue; // Skip empty arrays
        }

        num_vectors += 1;
        cspk_vectors.push(array);
    }

    let qvals: Array1<f64> = qvals.to_owned_array();
    let numqvals: usize = qvals.len();
    let q_reshaped: ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>> =
        qvals.into_shape((numqvals, 1, 1)).unwrap();

    // Assume calculate_spkd returns ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>
    let d: ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>> =
        calculate_spkd(&cspk_vectors, &q_reshaped, num_vectors, res);

    // Convert the ArrayBase to a PyArray
    let py_array: &numpy::PyArray<f64, Dim<[usize; 3]>> = d.into_pyarray(py);

    // Convert the PyArray to a PyObject
    Ok(py_array.to_object(py))
}

#[pymodule]
fn rs_distances(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(calculate_spkd_impl))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(iterate_spiketrains_impl))
        .unwrap();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use env_logger::Builder;
    use log::LevelFilter;
    use ndarray::{ArrayBase, Dim, OwnedRepr};
    use std::env;

    #[test]
    fn test_calculate_spkd() {
        env::set_var("RUST_LOG", "debug");
        Builder::new().filter_level(LevelFilter::Debug).init();

        // Mock data - list of 1D np.ndarrays, 3 spike trains
        let mut mock_data: Vec<ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>> = vec![
            ArrayBase::from(vec![0.1, 0.15, 0.2, 0.25, 0.3]),
            ArrayBase::from(vec![0.35, 0.4, 0.45, 0.5, 0.55]),
            ArrayBase::from(vec![0.6, 0.65, 0.7, 0.75, 0.8]),
        ];

        // simulate a view of data given from a numpy spike train
        let qvals = Array1::from(vec![1.0, 2.0, 3.0]);

        let numqvals = qvals.len();
        let q_reshaped = qvals.into_shape((numqvals, 1, 1)).unwrap();

        // calculate_spkd(&mut mock_data, &q_reshaped, Option::None);
    }
}
