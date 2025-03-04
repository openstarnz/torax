CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'Ti': {0.0: 10.0, 1.0: 11.0},
            'Ti_bound_right': 11.0,
            'Te': {0.0: 8.0, 1.0: 9.0},
            'Te_bound_right': 8.5,
            'ne_bound_right': 0.25,
            'ne': {0.0: 0.1, 1.0: 0.3},
            'ne_is_fGW': False,
            'normalize_to_nbar': False,
            'set_pedestal': False,
        },
        'numerics': {
            'ion_heat_eq': True,
            'el_heat_eq': True,
            'dens_eq': True,
            'current_eq': False,
            't_final': 100.0,
            'fixed_dt': 2.0,
        },
    },
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        # No sources
        'j_bootstrap': {'mode': 'zero'},
        'generic_current_source': {'mode': 'zero'},
        'generic_particle_source': {
            'S_tot': 10e20,
            'deposition_location': 0.75,
            'particle_width': 0.1,
        },
        # 'generic_particle_source': {'mode': 'zero'},
        'gas_puff_source': {'mode': 'zero'},
        'pellet_source': {'mode': 'zero'},
        'generic_ion_el_heat_source': {
            'rsource': 0.5,
            'w': 0.1,
            'Ptot': 40e6,
            'el_heat_fraction': 0.75,
        },
        # 'generic_ion_el_heat_source': {'mode': 'zero'},
        'fusion_heat_source': {'mode': 'zero'},
        'qei_source': {'mode': 'zero'},
        'ohmic_heat_source': {'mode': 'zero'},
    },
    'transport': {
        'transport_model': 'constant',
        'constant_params': {
            'chii_const': 1.0,
            'chie_const': 2.0,
            'De_const': 0.25,
            'Ve_const': 0.1,
            'chimin': 0.0,
            'Demin': 0.0,
            'Vemin': 0.0,
        },
    },
    'stepper': {
        'stepper_type': 'linear',
        'predictor_corrector': True,
        'corrector_steps': 1,
    },
    'time_step_calculator': {
        'calculator_type': 'fixed',
    },
}
