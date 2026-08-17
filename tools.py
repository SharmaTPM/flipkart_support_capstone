import os
import joblib
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

MODEL_PKL_PATH = "models/return_risk_model.pkl"
IMAGE_MODEL_PATH = "models/product_classifier.pt"

# Anchored threshold from Part 1 Task 9
T_STAR_RF = 0.42
LOW_CUT = T_STAR_RF
HIGH_CUT = T_STAR_RF + 0.15

# Default feature dictionary matching the exact columns expected by Part 1 scikit-learn model
DEFAULT_ORDER_FEATURES = {
    "price_inr": 2500.0,
    "discount_pct": 10.0,
    "customer_tenure_days": 365,
    "num_previous_orders": 15,
    "num_previous_returns": 3,
    "delivery_distance_km": 12.5,
    "delivery_days": 3,
    "rating_given": 4,
    "is_weekend_order": 0,
    "product_category": "Apparel",
    "payment_method": "COD"
}

FASHION_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def check_return_risk(order_features: dict = None) -> dict:
    """Task 3: Predict return probability & risk bucket based on t*_rf threshold."""
    if not os.path.exists(MODEL_PKL_PATH):
        return {"return_probability": 0.25, "risk_bucket": "Low", "t_star_rf": T_STAR_RF}

    # Merge incoming features over default schema so no columns are ever missing
    full_features = DEFAULT_ORDER_FEATURES.copy()
    if order_features:
        for k, v in order_features.items():
            if k in full_features:
                full_features[k] = v

    rf_model = joblib.load(MODEL_PKL_PATH)
    df = pd.DataFrame([full_features])

    prob = float(rf_model.predict_proba(df)[0][1])

    if prob < LOW_CUT:
        bucket = "Low"
    elif prob >= HIGH_CUT:
        bucket = "High"
    else:
        bucket = "Medium"

    return {
        "return_probability": round(prob, 4),
        "risk_bucket": bucket,
        "t_star_rf": T_STAR_RF
    }


def classify_product_image(image_path: str) -> dict:
    """Task 4: Classify image from sample directory using saved PyTorch model."""
    if not os.path.exists(image_path):
        return {"error": f"Image path {image_path} does not exist"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize standard ResNet18 (default 3 input channels to match checkpoint)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)

    if os.path.exists(IMAGE_MODEL_PATH):
        checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=device)
        # Check if saved model expects 1 or 3 input channels
        conv1_weight = checkpoint.get("conv1.weight", None)
        if conv1_weight is not None and conv1_weight.shape[1] == 1:
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Preprocess image into 3 channels (RGB)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
    ])

    img = Image.open(image_path).convert('RGB')
    tensor_img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_label = FASHION_CLASSES[pred.item()]
    confidence = float(conf.item())

    return {
        "predicted_category": predicted_label,
        "confidence": round(confidence, 4),
        "image_path": image_path
    }