from pydantic import BaseModel, Field


class HeartInput(BaseModel):
    """Input features from the UCI Heart Disease dataset."""

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 57, "sex": 1, "cp": 0, "trestbps": 140, "chol": 241,
                "fbs": 0, "restecg": 1, "thalach": 123, "exang": 1,
                "oldpeak": 0.2, "slope": 1, "ca": 0, "thal": 3,
            }
        }
    }


class PredictionOutput(BaseModel):
    heart_disease: bool
    probability: float = Field(..., description="Probability of heart disease (0-1)")


FEATURES = list(HeartInput.model_fields.keys())
