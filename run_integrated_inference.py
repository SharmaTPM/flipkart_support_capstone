import torch
import torch.nn as nn
import joblib
import pandas as pd
from PIL import Image, ImageOps
from torchvision import transforms, models

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Fashion-MNIST ResNet-18 Classifier Loader
def load_product_classifier(model_path="models/product_classifier.pt"):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # 10 Fashion-MNIST classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Preprocessing matching transfer-learning transforms
img_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Single Image Prediction Tool
def classify_product_image(image_path: str, model=None) -> dict:
    if model is None:
        model = load_product_classifier()
        
    image = Image.open(image_path).convert("L")
    tensor_img = img_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": round(probs[0, pred_idx].item(), 4)
    }

# 4. Integrated Claim Evaluation Engine
def evaluate_integrated_claim(order_data: dict, image_path: str, threshold: float = 0.44):
    tabular_model = joblib.load("models/return_risk_model.pkl")
    
    # Ensure correct data types for categorical encoding
    order_df = pd.DataFrame([order_data])
    risk_prob = tabular_model.predict_proba(order_df)[0, 1]
    is_high_risk = risk_prob >= threshold

    vision_result = classify_product_image(image_path)
    predicted_category = vision_result["predicted_class"]
    
    declared_category = order_data.get("product_category", "")
    category_mismatch = (
        declared_category.lower() != predicted_category.lower() 
        if declared_category else False
    )

    if is_high_risk:
        action = "ESCALATE TO HUMAN AGENT"
        reason = f"High tabular risk score ({risk_prob:.4f} >= {threshold})"
    elif category_mismatch:
        action = "ESCALATE TO HUMAN AGENT"
        reason = f"Category Mismatch (Declared: '{declared_category}' vs Detected: '{predicted_category}')"
    else:
        action = "AUTO-APPROVE RETURN & RESTOCK"
        reason = "Low tabular risk and verified product category match"

    return {
        "tabular_risk_prob": round(float(risk_prob), 4),
        "is_high_risk": bool(is_high_risk),
        "declared_category": declared_category,
        "predicted_category": predicted_category,
        "vision_confidence": vision_result["confidence"],
        "recommended_action": action,
        "routing_reason": reason
    }

# 5. Pipeline Test Driver
if __name__ == "__main__":
    import glob

    sample_images = glob.glob("data/sample_images/*.png")
    test_img = sample_images[0] if sample_images else "data/sample_images/00_ankle_boot.png"
    
    # Matching exact tabular features and types
    sample_order = {
        "price_inr": 1200.0,
        "discount_pct": 10.0,
        "delivery_days": 3,
        "customer_tenure_days": 180,
        "num_previous_orders": 12,
        "num_previous_returns": 0,
        "rating_given": 4,
        "delivery_distance_km": 15.0,
        "product_category": "Ankle boot",
        "payment_method": "Prepaid_Card",
        "is_weekend_order": 0
    }

    print("\n" + "=" * 60)
    print(f"Testing Single Image Tool on: {test_img}")
    print("=" * 60)
    print(classify_product_image(test_img))

    print("\n" + "=" * 60)
    print("Testing Integrated Multi-Modal Pipeline")
    print("=" * 60)
    res = evaluate_integrated_claim(sample_order, test_img)
    for k, v in res.items():
        print(f"{k:<25}: {v}")
    print("=" * 60)