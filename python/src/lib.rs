#[pyo3::pymodule]
mod PyRustNN {
  
  use pyo3::prelude::*;
  use pyo3::exceptions::PyValueError;
  use std::sync::Mutex;

  // Bring your types into scope
  use nn_core::deep_q_learning::deeq_q_network::{DeepQNetwork};
  use nn_core::general::activation::{ActivationType};

  /// Convert Python activation name -> ActivationType used by your crate.
  /// Add or adapt names to match your ActivationType enum.
  fn parse_activation(name: &str) -> Option<ActivationType> {
      match name.to_lowercase().as_str() {
          "relu" => Some(ActivationType::ReLU),
          "leaky_relu" => Some(ActivationType::LeakyReLU(0.1)),
          "sigmoid" => Some(ActivationType::Sigmoid),
          "tanh" => Some(ActivationType::Tanh),
          _ => None,
      }
  }

  #[pyclass]
  pub struct PyDeepQNetwork {
      // we use a Mutex so Python calls can borrow mutably
      // If your Network/Device are NOT Send + Sync, this may fail to compile.
      inner: Mutex<DeepQNetwork>,
  }

  #[pymethods]
  impl PyDeepQNetwork {
      /// Constructor: topology as list[int], activations as list[str].
      /// Example: topology=[8,64,4], activations=["relu","relu","linear"]
      #[new]
      pub fn new(py_topology: Vec<usize>, py_activations: Vec<String>, batch_size: usize) -> PyResult<Self> {
          // Convert topology -> Vec<u32>
          let topology: Vec<u32> = py_topology.iter().map(|&x| x as u32).collect();

          // Convert activation strings
          let mut activations: Vec<ActivationType> = Vec::new();
          for name in py_activations {
              match parse_activation(&name) {
                  Some(a) => activations.push(a),
                  None => return Err(PyValueError::new_err(format!("Unknown activation '{}'", name))),
              }
          }

          // Build network
          let dqn = DeepQNetwork::new(batch_size as u32, &topology, &activations);

          // If DeepQNetwork::new returned Result, adapt to handle errors.
          Ok(PyDeepQNetwork {
              inner: Mutex::new(dqn),
          })
      }

      /// choose_action: accept python list/sequence of floats
      pub fn choose_action(&self, py_state: Vec<f32>) -> PyResult<usize> {
          let mut guard = self.inner.lock().map_err(|_| PyValueError::new_err("Mutex poisoned"))?;
          let action = guard.choose_action(py_state);
          Ok(action as usize)
      }

      /// choose_best_action: deterministic choice
      pub fn choose_best_action(&self, py_state: Vec<f32>) -> PyResult<usize> {
          let guard = self.inner.lock().map_err(|_| PyValueError::new_err("Mutex poisoned"))?;
          let action = guard.choose_best_action(py_state);
          Ok(action as usize)
      }

      /// Add a single experience to the replay buffer
      /// `pre_state` and `observation` are python lists of floats
      pub fn add_experience(&self, pre_state: Vec<f32>, action: usize, reward: f32, finished: bool, observation: Vec<f32>) -> PyResult<()> {
          let mut guard = self.inner.lock().map_err(|_| PyValueError::new_err("Mutex poisoned"))?;
          // push to internal replay_buffer (the tuple type in your code: Vec<(Vec<f32>, u32, f32, bool, Vec<f32>)>)
          guard.replay_buffer.push_back((pre_state, action as u32, reward, finished, observation));
          Ok(())
      }

      pub fn train(&self) -> PyResult<()> {
          let mut guard = self.inner.lock().map_err(|_| PyValueError::new_err("Mutex poisoned"))?;
          // If your train expects &mut self (recommended), it will run here.
          guard.train();
          Ok(())
      }

      pub fn update_target(&mut self) -> PyResult<()>  {
        let mut guard = self.inner.lock().map_err(|_| PyValueError::new_err("Mutex poisoned"))?;
          guard.update_target();
          Ok(())
      } 

      pub fn save_data(&self, path: String) -> PyResult<()> {
        let guard = self.inner.lock().map_err(|_| PyValueError::new_err("Mutex poisoned"))?;
        guard.save_to_onnx(path);
          Ok(())
            
      }

      /// Optional: expose method to set epsilon from Python
      pub fn set_epsilon(&self, new_eps: f32) -> PyResult<()> {
          let mut guard = self.inner.lock().map_err(|_| PyValueError::new_err("Mutex poisoned"))?;
          guard.epsilon = new_eps;
          Ok(())
      }

      /// Optional: get some diagnostic (e.g. replay buffer size)
      pub fn replay_len(&self) -> PyResult<usize> {
          let guard = self.inner.lock().map_err(|_| PyValueError::new_err("Mutex poisoned"))?;
          Ok(guard.replay_buffer.len())
      }
  }

}