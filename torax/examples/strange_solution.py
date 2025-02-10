CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'ne_bound_right': 1.0,
            'ne': {0.0: 1.5, 1.0: 1.0},
            'set_pedestal': False,
        },
        # Only solve the density equation
        'numerics': {
            'ion_heat_eq': False,
            'el_heat_eq': False,
            'current_eq': False,
            'dens_eq': True,
            't_final': 5.0,
            'fixed_dt': 0.1,
        },
    },
    # The default circular geometry
    'geometry': {
        'geometry_type': 'circular',
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
    # Set constant transport coefficients everywhere
    'transport': {
        'transport_model': 'constant',
        'constant_params': {
            'chii_const': 1.0,
            'chie_const': 1.0,
            'De_const': 1.0,
            'Ve_const': 1.0,
            'chimin': 0.0,
            'Demin': 0.0,
            'Vemin': 0.0,
        },
    },
    'stepper': {
        'stepper_type': 'linear',
    },
    'time_step_calculator': {
        'calculator_type': 'fixed',
    },
}