"""
BRFSS 2022 Script para obtener el Dataset
Se replica el procesado del autor Kamil Pytlak's notebook y se agregan columnas adicionales
https://github.com/kamilpytlak/data-science-projects/blob/main/heart-disease-prediction/2022/notebooks/data_processing.ipynb
"""

from pathlib import Path
import numpy as np
import pandas as pd
import zipfile

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DATA_PATH = Path('./data')
RAW_DATA_PATH = DATA_PATH / 'raw'
PROCESSED_DATA_PATH = DATA_PATH

# Create directories
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

RAW_FILE_PATH = DATA_PATH / 'LLCP2022XPT.zip'
FINAL_FILE_WITH_NANS_PATH = PROCESSED_DATA_PATH / '01_heart_with_nans.csv'

# ============================================================================
# COLUMN ORGANIZATION BY TYPE
# ============================================================================

# Survey Information
SURVEY_COLS = [
    ("IDATE", "SurveyDate"),
    ("DISPCODE", "DispositionCode"),
]

# Demographics
DEMOGRAPHICS_COLS = [
    ("_STATE", "State"),
    ("SEXVAR", "Sex"),
]

# General Health
GENERAL_HEALTH_COLS = [
    ("GENHLTH", "GeneralHealth"),
    ("PHYSHLTH", "PhysicalHealthDays"),
    ("MENTHLTH", "MentalHealthDays"),
    ("POORHLTH", "PoorHealthDays"),
    ("SLEPTIM1", "SleepHours"),
]

# Healthcare Access
HEALTHCARE_COLS = [
    ("CHECKUP1", "LastCheckupTime"),
    ("LASTDEN4", "LastDentistVisit"),
]

# Cardiovascular Diseases
CARDIOVASCULAR_COLS = [
    ("CVDINFR4", "HadHeartAttack"),
    ("CVDCRHD4", "HadAngina"),
    ("CVDSTRK3", "HadStroke"),
]

# Chronic Diseases
CHRONIC_DISEASE_COLS = [
    ("ASTHMA3", "HadAsthma"),
    ("CHCSCNC1", "HadSkinCancer"),
    ("CHCCOPD3", "HadCOPD"),
    ("ADDEPEV3", "HadDepressiveDisorder"),
    ("CHCKDNY2", "HadKidneyDisease"),
    ("HAVARTH4", "HadArthritis"),
    ("DIABETE4", "HadDiabetes"),
]

# Disabilities
DISABILITIES_COLS = [
    ("DEAF", "DeafOrHardOfHearing"),
    ("BLIND", "BlindOrVisionDifficulty"),
    ("DECIDE", "DifficultyConcentrating"),
    ("DIFFWALK", "DifficultyWalking"),
    ("DIFFDRES", "DifficultyDressingBathing"),
    ("DIFFALON", "DifficultyErrands"),
]

# Oral Health
ORAL_HEALTH_COLS = [
    ("RMVTETH4", "RemovedTeeth"),
]

# Physical Activity
PHYSICAL_ACTIVITY_COLS = [
    ("EXERANY2", "PhysicalActivities"),
]

# Tobacco Use
TOBACCO_COLS = [
    ("_SMOKER3", "SmokerStatus"),
    ("SMOKE100", "Ever100Cigarettes"),
    ("SMOKDAY2", "SmokeFrequency"),
    ("LCSFIRST", "AgeFirstCigarette"),
    ("LCSNUMCG", "CigarettesPerDay"),
    ("LCSSCNCR", "LungCancerConcern"),
    ("ECIGNOW2", "ECigaretteUsage"),
    ("LCSCTSC1", "ChestScan"),
]

# Alcohol Consumption
ALCOHOL_COLS = [
    ("_DRNKWK2", "DrinksPerWeek"),
    ("_RFDRHV8", "AlcoholRiskIndicator"),
    ("DRNKANY6", "AlcoholDrinkers"),
    
    ('ALCDAY4','AlcoholConsumptionDays'),
    ('AVEDRNK3', 'AverageDrinksPerDay'),
    ('DRNK3GE5', 'DaysWithFivePlusDrinks'),
    ('MAXDRNKS', 'MaxDrinksPerOccasion'),
    ('_RFBING6', 'BingeIndicator'),
    ('_DRNKWK2', 'DrinksPerWeek'),
    ('_RFDRHV8', 'AlcoholRiskIndicator'),
    ('ASBIALCH', 'AlcoholAdvice'),
    ('ASBIDRNK', 'AlcoholConsumption'),
    ('ASBIBING', 'BingeConsumption'),
    ('ASBIADVC', 'BingeAdvice'),
    ('ASBIRDUC', 'ReduceConsumption'),
]

# Anthropometric Data
ANTHROPOMETRIC_COLS = [
    ("HTM4", "HeightInMeters"),
    ("WTKG3", "WeightInKilograms"),
    ("_BMI5", "BMI"),
]

# Race/Ethnicity and Age
RACE_AGE_COLS = [
    ("_RACEGR4", "RaceEthnicityCategory"),
    ("_AGEG5YR", "AgeCategory"),
]

# Vaccinations
VACCINATIONS_COLS = [
    ("FLUSHOT7", "FluVaxLast12"),
    ("PNEUVAC4", "PneumoVaxEver"),
    ("TETANUS1", "TetanusLast10Tdap"),
]

# Health Tests
HEALTH_TESTS_COLS = [
    ("_AIDTST4", "HIVTesting"),
    ("HIVRISK5", "HighRiskLastYear"),
    ("COVIDPOS", "CovidPos"),
]

# Combine all columns in order
ALL_COLUMNS = (
    SURVEY_COLS +
    DEMOGRAPHICS_COLS +
    GENERAL_HEALTH_COLS +
    HEALTHCARE_COLS +
    CARDIOVASCULAR_COLS +
    CHRONIC_DISEASE_COLS +
    DISABILITIES_COLS +
    ORAL_HEALTH_COLS +
    PHYSICAL_ACTIVITY_COLS +
    TOBACCO_COLS +
    ALCOHOL_COLS +
    ANTHROPOMETRIC_COLS +
    RACE_AGE_COLS +
    VACCINATIONS_COLS +
    HEALTH_TESTS_COLS
)

ORIGINAL_VAR_NAMES = [col[0] for col in ALL_COLUMNS]
NEW_VAR_NAMES = [col[1] for col in ALL_COLUMNS]

# ============================================================================
# VALUE MAPPINGS (FROM KAMIL'S NOTEBOOK)
# ============================================================================

STATE = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "District of Columbia",
    12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois",
    18: "Indiana", 19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana",
    23: "Maine", 24: "Maryland", 25: "Massachusetts", 26: "Michigan",
    27: "Minnesota", 28: "Mississippi", 29: "Missouri", 30: "Montana",
    31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey",
    35: "New Mexico", 36: "New York", 37: "North Carolina", 38: "North Dakota",
    39: "Ohio", 40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania",
    44: "Rhode Island", 45: "South Carolina", 46: "South Dakota", 47: "Tennessee",
    48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia", 53: "Washington",
    54: "West Virginia", 55: "Wisconsin", 56: "Wyoming", 66: "Guam",
    72: "Puerto Rico", 78: "Virgin Islands"
}

SEX = {1: 'Male', 2: 'Female'}

GEN_HEALTH = {
    1: "Excellent",
    2: "Very good",
    3: "Good",
    4: "Fair",
    5: "Poor"
}

PHYS_MEN_HEALTH = {77: np.nan, 88: 0, 99: np.nan}

LAST_CHECKUP = {
    1: "Within past year (anytime less than 12 months ago)",
    2: "Within past 2 years (1 year but less than 2 years ago)",
    3: "Within past 5 years (2 years but less than 5 years ago)",
    4: "5 or more years ago"
}

YES_NO_QUESTIONS = {1: 'Yes', 2: 'No'}

SLEEP_TIME = lambda x: np.where(x > 24, np.nan, x)

TEETH_REMOVED = {
    1: "1 to 5",
    2: "6 or more, but not all",
    3: "All",
    8: "None of them"
}

DIABETES = {
    1: "Yes",
    2: "Yes, but only during pregnancy (female)",
    3: "No",
    4: "No, pre-diabetes or borderline diabetes",
}

SMOKER_STATUS = {
    1: "Current smoker - now smokes every day",
    2: "Current smoker - now smokes some days",
    3: "Former smoker",
    4: "Never smoked"
}

ECIGARETTES = {
    1: "Never used e-cigarettes in my entire life",
    2: "Use them every day",
    3: "Use them some days",
    4: "Not at all (right now)"
}

RACE = {
    1: "White only, Non-Hispanic",
    2: "Black only, Non-Hispanic",
    3: "Other race only, Non-Hispanic",
    4: "Multiracial, Non-Hispanic",
    5: "Hispanic"
}

AGE_CATEGORY = {
    1: "Age 18 to 24", 2: "Age 25 to 29", 3: "Age 30 to 34", 4: "Age 35 to 39",
    5: "Age 40 to 44", 6: "Age 45 to 49", 7: "Age 50 to 54", 8: "Age 55 to 59",
    9: "Age 60 to 64", 10: "Age 65 to 69", 11: "Age 70 to 74", 12: "Age 75 to 79",
    13: "Age 80 or older"
}

TETANUS = {
    1: "Yes, received Tdap",
    2: "Yes, received tetanus shot, but not Tdap",
    3: "Yes, received tetanus shot but not sure what type",
    4: "No, did not receive any tetanus shot in the past 10 years",
}

COVID = {
    1: "Yes",
    2: "No",
    3: "Tested positive using home test without a health professional"
}

DISPCODE_MAP = {
    1100: "Completed interview",
    1200: "Partially completed interview",
    1300: "Not eligible",
    2100: "Refused",
    2200: "Respondent ill or deceased",
    2300: "Language barrier",
    2400: "Physically or mentally unable",
    2500: "Other non-response",
}

SMOKE_100_MAP = {1: "Yes", 2: "No", 7: "Don't know", 9: "Refused"}

SMOKE_FREQUENCY_MAP = {1: "Every day", 2: "Some days", 3: "Not at all"}

LAST_DENTIST_MAP = {
    1: "Less than 1 year ago",
    2: "1 year to less than 2 years ago",
    3: "2 years to less than 5 years ago",
    4: "5 years or more",
    7: "Don't know",
    9: "Refused"
}

# ============================================================================
# CONVERSION FUNCTIONS
# ============================================================================

def extract_sas_data():
    """Extract SAS XPT file from local zip if not already extracted"""
    
    print("Extracting SAS XPT file from local zip...")
    
    if not RAW_FILE_PATH.exists():
        print(f"✗ File not found: {RAW_FILE_PATH}")
        return False
    
    try:
        with zipfile.ZipFile(RAW_FILE_PATH, 'r') as zip_ref:
            # 1. Obtiene la lista de archivos dentro del zip
            zip_contents = zip_ref.namelist()
            
            # 2. Verifica si TODOS los archivos del zip ya existen en el destino
            already_extracted = all((RAW_DATA_PATH / file_name).exists() for file_name in zip_contents)
            
            if already_extracted:
                print("--- Data already extracted. Skipping decompression. ---")
                return True
            
            # 3. Si falta alguno o ninguno está, se descomprime
            zip_ref.extractall(RAW_DATA_PATH)
            print("✓ Extraction completed")
            return True
            
    except Exception as e:
        print(f"✗ Error extracting data: {e}")
        return False
    
    
def read_sas_data():
    """Read SAS XPT file"""
    print("Reading SAS XPT file...")
    
    xpt_files = list(RAW_DATA_PATH.glob('*.XPT'))
    if not xpt_files:
        print("✗ No .XPT file found")
        return None
    
    try:
        df = pd.read_sas(str(xpt_files[0]), encoding='utf-8')
        print(f"✓ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"✗ Error reading SAS file: {e}")
        return None

def select_and_rename_columns(df):
    """Select columns and rename them, maintaining order"""
    print(f"Selecting {len(ORIGINAL_VAR_NAMES)} columns...")
    
    # Select only available columns
    available_vars = [v for v in ORIGINAL_VAR_NAMES if v in df.columns]
    df_selected = df[available_vars].copy()
    
    # Create mapping for available columns only
    var_mapping = {orig: new for orig, new in zip(ORIGINAL_VAR_NAMES, NEW_VAR_NAMES) if orig in available_vars}
    df_selected.rename(columns=var_mapping, inplace=True)
    
    print(f"✓ Selected {df_selected.shape[1]} columns")
    return df_selected

def apply_mappings(df):
    """Apply value mappings to columns"""
    print("Applying value mappings...")
    
    df_copy = df.copy()
    
    # Apply mappings
    if 'State' in df_copy.columns:
        df_copy['State'] = df_copy['State'].map(STATE)
    if 'Sex' in df_copy.columns:
        df_copy['Sex'] = df_copy['Sex'].map(SEX)
    if 'GeneralHealth' in df_copy.columns:
        df_copy['GeneralHealth'] = df_copy['GeneralHealth'].map(GEN_HEALTH)
    if 'PhysicalHealthDays' in df_copy.columns:
        df_copy['PhysicalHealthDays'] = df_copy['PhysicalHealthDays'].replace(PHYS_MEN_HEALTH)
    if 'MentalHealthDays' in df_copy.columns:
        df_copy['MentalHealthDays'] = df_copy['MentalHealthDays'].replace(PHYS_MEN_HEALTH)
    if 'PoorHealthDays' in df_copy.columns:
        df_copy['PoorHealthDays'] = df_copy['PoorHealthDays'].replace(PHYS_MEN_HEALTH)
    if 'LastCheckupTime' in df_copy.columns:
        df_copy['LastCheckupTime'] = df_copy['LastCheckupTime'].map(LAST_CHECKUP)
    if 'PhysicalActivities' in df_copy.columns:
        df_copy['PhysicalActivities'] = df_copy['PhysicalActivities'].map(YES_NO_QUESTIONS)
    if 'SleepHours' in df_copy.columns:
        df_copy['SleepHours'] = df_copy['SleepHours'].apply(SLEEP_TIME)
    if 'RemovedTeeth' in df_copy.columns:
        df_copy['RemovedTeeth'] = df_copy['RemovedTeeth'].map(TEETH_REMOVED)
    if 'HadDiabetes' in df_copy.columns:
        df_copy['HadDiabetes'] = df_copy['HadDiabetes'].map(DIABETES)
    if 'SmokerStatus' in df_copy.columns:
        df_copy['SmokerStatus'] = df_copy['SmokerStatus'].map(SMOKER_STATUS)
    if 'Ever100Cigarettes' in df_copy.columns:
        df_copy['Ever100Cigarettes'] = df_copy['Ever100Cigarettes'].map(SMOKE_100_MAP)
    if 'SmokeFrequency' in df_copy.columns:
        df_copy['SmokeFrequency'] = df_copy['SmokeFrequency'].map(SMOKE_FREQUENCY_MAP)
    if 'ECigaretteUsage' in df_copy.columns:
        df_copy['ECigaretteUsage'] = df_copy['ECigaretteUsage'].map(ECIGARETTES)
    if 'RaceEthnicityCategory' in df_copy.columns:
        df_copy['RaceEthnicityCategory'] = df_copy['RaceEthnicityCategory'].map(RACE)
    if 'AgeCategory' in df_copy.columns:
        df_copy['AgeCategory'] = df_copy['AgeCategory'].map(AGE_CATEGORY)
    if 'TetanusLast10Tdap' in df_copy.columns:
        df_copy['TetanusLast10Tdap'] = df_copy['TetanusLast10Tdap'].map(TETANUS)
    if 'CovidPos' in df_copy.columns:
        df_copy['CovidPos'] = df_copy['CovidPos'].map(COVID)
    if 'DispositionCode' in df_copy.columns:
        df_copy['DispositionCode'] = df_copy['DispositionCode'].map(DISPCODE_MAP)
    if 'LastDentistVisit' in df_copy.columns:
        df_copy['LastDentistVisit'] = df_copy['LastDentistVisit'].map(LAST_DENTIST_MAP)
    
    # Map Yes/No columns
    yes_no_cols = [
        'HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer',
        'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
        'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating',
        'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands',
        'ChestScan', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
        'HighRiskLastYear', 'LungCancerConcern', 'AlcoholConsumption',
        'BingeConsumption'
    ]
    
    for col in yes_no_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].map(YES_NO_QUESTIONS)
    
    # Numeric conversions (from Kamil's notebook)
    if 'HeightInMeters' in df_copy.columns:
        df_copy['HeightInMeters'] = df_copy['HeightInMeters'] / 100
    if 'WeightInKilograms' in df_copy.columns:
        df_copy['WeightInKilograms'] = df_copy['WeightInKilograms'] / 100
    if 'BMI' in df_copy.columns:
        df_copy['BMI'] = df_copy['BMI'] / 100
    
    print("✓ Mappings applied")
    return df_copy

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("BRFSS 2022 DATA PROCESSING - KAMIL'S METHOD (ORGANIZED BY TYPE)")
    print("="*80)
    
    # Step 1: Extract
    print("\n[1/4] Extracting data...")
    if not extract_sas_data():
        print("Aborting...")
        return
    
    # Step 2: Read
    print("\n[2/4] Reading data...")
    df = read_sas_data()
    if df is None:
        print("Aborting...")
        return
    
    # Step 3: Select and rename
    print("\n[3/4] Selecting and renaming columns...")
    df_selected = select_and_rename_columns(df)
    
    # Step 4: Apply mappings
    print("\n[4/4] Applying mappings...")
    df_processed = apply_mappings(df_selected)
    
    # Export
    print("\nExporting data...")
    df_processed.to_csv(FINAL_FILE_WITH_NANS_PATH, index=False)
    print(f"✓ Exported (with NaNs): {FINAL_FILE_WITH_NANS_PATH}")
    
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETED")
    print("="*80)
    print(f"Dataset shape: {df_processed.shape[0]} rows × {df_processed.shape[1]} columns")
    print(f"\nColumn organization:")
    print(f"  - Survey Information: 2 columns")
    print(f"  - Demographics: 2 columns")
    print(f"  - General Health: 5 columns")
    print(f"  - Healthcare Access: 2 columns")
    print(f"  - Cardiovascular Diseases: 3 columns")
    print(f"  - Chronic Diseases: 7 columns")
    print(f"  - Disabilities: 6 columns")
    print(f"  - Oral Health: 1 column")
    print(f"  - Physical Activity: 1 column")
    print(f"  - Tobacco Use: 8 columns")
    print(f"  - Alcohol Consumption: 13 columns")
    print(f"  - Anthropometric Data: 3 columns")
    print(f"  - Race/Ethnicity & Age: 2 columns")
    print(f"  - Vaccinations: 3 columns")
    print(f"  - Health Tests: 3 columns")
    print(f"\nTotal: {df_processed.shape[1]} columns")
    print(f"\nFiles saved in: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
