import os
import torch

from torchvision import models, transforms
import torch.nn as nn

from PIL import Image


class DisasterPredictor:

    def __init__(self):

        print("Initializing Disaster AI...")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = None

        model_path = os.path.join(
            os.path.dirname(__file__),
            "flood_model.pth"
        )


        # Check if trained model exists

        if os.path.exists(model_path):

            print("Loading trained model...")


            self.model = models.resnet18(
                weights=None
            )


            self.model.fc = nn.Linear(
                self.model.fc.in_features,
                2
            )


            self.model.load_state_dict(
                torch.load(
                    model_path,
                    map_location=self.device
                )
            )


            self.model.to(
                self.device
            )


            self.model.eval()


            print(
                "Model loaded successfully"
            )


        else:

            print(
                "No trained model found. Running demo mode."
            )



        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224,224)
                ),

                transforms.ToTensor()
            ]
        )



    def predict(self, image_path):


        # Demo mode

        if self.model is None:

            return {

                "disaster": "Flood",

                "confidence": 95.5,

                "area": "Not calculated",

                "severity": "Pending"

            }



        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        image = self.transform(
            image
        )


        image = image.unsqueeze(0)

        image = image.to(
            self.device
        )


        with torch.no_grad():

            output = self.model(
                image
            )


            probability = torch.softmax(
                output,
                dim=1
            )


            confidence, prediction = torch.max(
                probability,
                1
            )


        classes = [
            "Flood",
            "No Flood"
        ]


        return {

            "disaster": classes[prediction.item()],

            "confidence": round(
                confidence.item()*100,
                2
            ),

            "area": "Not calculated",

            "severity": "Pending"

        }