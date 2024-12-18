"""
Configuration to demonstrate unphysical results when flux surfaces move.
If the number density and temperature are initially set to be constant across rhon,
and all transport and sources are disabled, the solution should remain constant,
even if the flux surfaces move. However, this example demonstrates that varying B0 with time
causes a noticeable effect on the solution, which is unphysical.
"""

START_B0 = 1.0
MID_B0 = 2.0

CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'set_pedestal': False,
            # A constant inital Ti, Te and ne
            'Ti': {0: 15.0, 1: 15.0},
            'Te': {0: 15.0, 1: 15.0},
            'ne': {0: 1.5, 1: 1.5},
        },
        'plasma_composition': {},
        'numerics': {
            'ion_heat_eq': True,
            'el_heat_eq': True,
            'current_eq': False,
            'dens_eq': True,
            # Disabling Phibdot calculation will leave Ti, Te and ne fixed,
            #  demonstrating that the erronous effect is due to Phibdot
            # 'calcphibdot': False,
        },
    },
    'geometry': {
        'geometry_type': 'circular',
        'n_rho': 25,
        'geometry_configs': {
          # Create a triangular ramp of B0.
          # For the first second, B0 will be START_B0,
          # then will ramp to and from MID_B0 over the course of 2 seconds
          0.0: {
            'B0': START_B0,
          },
          1.0: {
            'B0': START_B0,
          },
          2.0: {
            'B0': MID_B0,
          },
          3.0: {
            'B0': START_B0,
          },
        }
    },
    # No sources
    'sources': {
        'j_bootstrap': {'mode': 'zero'},
        'generic_current_source': {'mode': 'zero'},
        'generic_particle_source': {'mode': 'zero'},
        'gas_puff_source': {'mode': 'zero'},
        'pellet_source': {'mode': 'zero'},
        'generic_ion_el_heat_source': {'mode': 'zero'},
        'fusion_heat_source': {'mode': 'zero'},
        'qei_source': {'mode': 'zero'},
        'ohmic_heat_source': {'mode': 'zero'},
    },
    # No transport
    'transport': {
        'transport_model': 'constant',
        'constant_params': {
            'chii_const': 0.0,
            'chie_const': 0.0,
            'De_const': 0.0,
            'Ve_const': 0.0,
            'chimin': 0.0,
            'Demin': 0.0,
            'Vemin': 0.0,
        },
    },
    # Simple stepper and time step calculator
    'stepper': {
        'stepper_type': 'linear',
    },
    'time_step_calculator': {
        'calculator_type': 'fixed',
    },
}
