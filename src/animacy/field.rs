//! Entity neural field state.

/// Entity neural field state
/// Implements: ∂a_E/∂t = -a_E + μ_E·a_E + inputs + lateral
#[derive(Debug, Clone)]
pub struct EntityField {
    /// Field activity over (form × motion_class)
    /// Simplified to discrete grid
    activity: Vec<Vec<f64>>,

    /// Number of form bins
    n_forms: usize,

    /// Number of motion class bins
    n_motion: usize,

    /// Gain parameter (μ_E)
    gain: f64,

    /// Decay rate
    decay: f64,

    /// Lateral kernel (winner-take-all)
    lateral_inhibition: f64,

    /// Input integration rate
    input_rate: f64,
}

impl EntityField {
    pub fn new(n_forms: usize, n_motion: usize, gain: f64) -> Self {
        Self {
            activity: vec![vec![0.0; n_motion]; n_forms],
            n_forms,
            n_motion,
            gain,
            decay: 0.1,
            lateral_inhibition: 0.5,
            input_rate: 0.3,
        }
    }

    /// Sigmoid nonlinearity
    fn sigma(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Update field with inputs from V4 and MT
    pub fn update(&mut self, v4_input: &[f64], mt_input: &[f64], dt: f64) {
        let mut new_activity = self.activity.clone();

        for i in 0..self.n_forms {
            for j in 0..self.n_motion {
                let a = self.activity[i][j];

                // Self-excitation with gain
                let self_term = -a + self.gain * a;

                // V4 form input
                let v4_term = if i < v4_input.len() {
                    self.input_rate * Self::sigma(v4_input[i])
                } else {
                    0.0
                };

                // MT motion input
                let mt_term = if j < mt_input.len() {
                    self.input_rate * Self::sigma(mt_input[j])
                } else {
                    0.0
                };

                // Lateral inhibition (winner-take-all)
                let total_activity: f64 = self.activity.iter().flatten().sum();
                let lateral_term = -self.lateral_inhibition * (total_activity - a);

                // Euler update
                let da = self_term + v4_term + mt_term + lateral_term;
                new_activity[i][j] = (a + da * dt).clamp(0.0, 1.0);
            }
        }

        self.activity = new_activity;
    }

    /// Get peak activity
    pub fn peak_activity(&self) -> (usize, usize, f64) {
        let mut max_val = 0.0;
        let mut max_i = 0;
        let mut max_j = 0;

        for (i, row) in self.activity.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_i = i;
                    max_j = j;
                }
            }
        }

        (max_i, max_j, max_val)
    }

    /// Total activity (for monitoring)
    pub fn total_activity(&self) -> f64 {
        self.activity.iter().flatten().sum()
    }

    /// Set gain (for psychedelic simulation)
    pub fn set_gain(&mut self, gain: f64) {
        self.gain = gain;
    }

    /// Get current gain
    pub fn gain(&self) -> f64 {
        self.gain
    }

    /// Set decay rate
    pub fn set_decay(&mut self, decay: f64) {
        self.decay = decay.clamp(0.01, 1.0);
    }

    /// Get current decay rate
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Apply decay to all activity (for natural activity reduction over time)
    pub fn apply_decay(&mut self, dt: f64) {
        let decay_factor = (-self.decay * dt).exp();
        for row in &mut self.activity {
            for val in row.iter_mut() {
                *val *= decay_factor;
            }
        }
    }
}
