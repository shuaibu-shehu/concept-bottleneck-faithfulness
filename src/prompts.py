"""The zero-shot checklist prompts: one present/absent pair per 7-point checklist criterion.

Used by encode_features.py to score the concepts and by tabulate_concepts.py for Table 3.
"""

CONCEPT_PROMPTS = {
    'atypical_pigment_network': ('a skin lesion with an atypical pigment network',
                                 'a skin lesion with a typical pigment network'),
    'blue_whitish_veil':        ('a skin lesion with a blue-whitish veil',
                                 'a skin lesion without a blue-whitish veil'),
    'atypical_vascular':        ('a skin lesion with atypical vascular structures',
                                 'a skin lesion with regular vascular structures'),
    'irregular_streaks':        ('a skin lesion with irregular streaks',
                                 'a skin lesion without irregular streaks'),
    'irregular_pigmentation':   ('a skin lesion with irregular pigmentation',
                                 'a skin lesion with regular pigmentation'),
    'irregular_dots_globules':  ('a skin lesion with irregular dots and globules',
                                 'a skin lesion with regular dots and globules'),
    'regression':               ('a skin lesion with regression structures',
                                 'a skin lesion without regression structures'),
}
