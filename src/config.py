drug_col = [
    'Amikacin',
    'Amoxicillin', 'Ampicillin', 'Azithromycin', 'Aztreonam', 
    # 'Cefadroxil',
    'Cefazolin', 'Cefditoren', 'Cefepime', 'Cefixime', 'Cefotaxime',
    'Cefotetan', 'Cefpodoxime', 'Ceftazidime', 'Ceftriaxone', 'Cefuroxime',
    'Cephradine', 'Cilastatin', 'Ciprofloxacin', 'Clarithromycin',
    'Clavulanate', 'Clindamycin', 'Doripenem', 'Doxycycline', 'Ertapenem',
    'Gentamicin', 'Imipenem', 'Levofloxacin', 'Linezolid', 'Meropenem',
    'Minocycline', 'Moxifloxacin', 'Nafcillin', 'Neomycin',
    'Penicillin', 'Piperacillin', 'Streptomycin', 'Sulbactam',
    'Sulfamethoxazole', 'Teicoplanin', 'Tetracycline', 'Tigecycline',
    'Tobramycin', 'tazobactam', 'trimethoprim'
]

# lab_cols = [
#     'ALP', 'ALT', 'ANC', 'AST', 'BT', 'BUN', 'Bilirubin', 'CRP', 'Cl', 'Cr',
#     'DBP', 'HR', 'Hb', 'K', 'LDH', 'Na', 'Neutrophil', 'PLT', 'PT_INR',
#     'PT_perc', 'PT_sec', 'RR', 'SBP', 'TCO2', 'WBC', 'aPTT'
# ]


lab_cols = [
    'SBP', 'DBP', 'HR', 'RR', 'BT', 'WBC', 'Hb', 'PLT', 'Neutrophil', 'ANC', 'albumin', 'protein',
    'Bilirubin', 'AST', 'ALP', 'ALT', 'BUN', 'Cr', 'Na', 'K', 'Cl', 'TCO2', 'CRP'
]

demo_col = [
    'age', 'sex'
]

lab_demo_col = lab_cols + demo_col

labnames = [
    'WBC', 'Hb', 'SBP', 'ALT', 'CRP', 'DBP', 
    'ANC', 'BUN', 'AST', 'Cl', 'TCO2', 'Cr',
    'Neutrophil', 'aPTT', 'Na', 'BT', 'PT_INR', 'LDH',
    'K', 'Bilirubin', 'RR', 'PLT', 'HR', 'PT_perc',
    'PT_sec', 'ALP', 'DBP', 'SBP', 'SBP', 'DBP']

blood = [
    'WBC', 'Hb', 'ALT', 'CRP', 
    'ANC', 'BUN', 'AST', 'Cl', 'TCO2', 'Cr',
    'Neutrophil', 'aPTT', 'Na', 'PT_INR', 'LDH',
    'K', 'Bilirubin', 'PLT', 'PT_perc',
    'PT_sec', 'ALP']

vital = [
    'SBP', 'DBP', 'RR', 'BT', 'HR'
]
labdict = {
    3000905: 'WBC',
    3000963: 'Hb',
    3004249: 'SBP',
    3006923: 'ALT',
    3010156: 'CRP',
    3012888: 'DBP',
    3013650: 'ANC', # /㎕
    3013682: 'BUN',
    3013721: 'AST',
    3014576: 'Cl',
    3015632: 'TCO2',
    3016723: 'Cr',
    3017354: 'Neutrophil', # %
    3018677: 'aPTT',
    3019550: 'Na',
    3020891: 'BT',
    3022217: 'PT_INR',
    3022250: 'LDH',
    3023103: 'K',
    3024128: 'Bilirubin',
    3024171: 'RR',
    3024929: 'PLT',
    3027018: 'HR',
    3033658: 'PT_perc',
    3034426: 'PT_sec',
    3035995: 'ALP',
    21490851: 'DBP',
    21490853: 'SBP',
    21492239: 'SBP',
    21492240: 'DBP',
    3007591: 'Neutrophil',
    3032080: 'PT_INR',
}

ccidict = {
    '만성폐질환': 'pulmonary',
    '악성종양': 'tumor', 
    '말초혈관질환': 'peripheral', 
    '신장병': 'kidney', 
    '만성합병증없는당뇨병': 'uncomplicated_diabetes', 
    '경증간질환': 'mild_liver', 
    '뇌혈관질환': 'cerebrovascular', 
    '전이성암': 'metastatic',
    '울혈성심부전': 'HF', 
    '만성합병증동반당뇨병': 'complicated_diabetes', 
    '결합조직병': 'tissue', 
    '치매': 'dimentia', 
    '소화성궤양': 'ulcer', 
    '심근경색증': 'MI', 
    '간질환': 'liver', 
    '마비': 'hemiplegia',
    'HIV': 'HIV'
}
