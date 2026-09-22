# api/schemas.py

from pydantic import BaseModel, Field
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class ChurnPredictionRequest(BaseModel):
    Gender: Literal["Male", "Female"]
    Senior_Citizen: Literal["No", "Yes"] = Field(alias="Senior Citizen")
    Partner: Literal["No", "Yes"]
    Dependents: Literal["No", "Yes"]
    Tenure_Months: int = Field(alias="Tenure Months", ge=0, le=100)
    Phone_Service: Literal["Yes", "No"] = Field(alias="Phone Service")
    Multiple_Lines: Literal["No", "Yes", "No phone service"] = Field(alias="Multiple Lines")
    Internet_Service: Literal["DSL", "Fiber optic", "No"] = Field(alias="Internet Service")
    Online_Security: Literal["Yes", "No", "No internet service"] = Field(alias="Online Security")
    Online_Backup: Literal["Yes", "No", "No internet service"] = Field(alias="Online Backup")
    Device_Protection: Literal["No", "Yes", "No internet service"] = Field(alias="Device Protection")
    Tech_Support: Literal["No", "Yes", "No internet service"] = Field(alias="Tech Support")
    Streaming_TV: Literal["No", "Yes", "No internet service"] = Field(alias="Streaming TV")
    Streaming_Movies: Literal["No", "Yes", "No internet service"] = Field(alias="Streaming Movies")
    Contract: Literal["Month-to-month", "Two year", "One year"]
    Paperless_Billing: Literal["Yes", "No"] = Field(alias="Paperless Billing")
    Payment_Method: Literal[
        "Mailed check", "Electronic check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(alias="Payment Method")
    Monthly_Charges: float = Field(alias="Monthly Charges", ge=0)
    Total_Charges: float = Field(alias="Total Charges", ge=0)

    model_config = ConfigDict(populate_by_name=True)


class ChurnPredictionResponse(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float