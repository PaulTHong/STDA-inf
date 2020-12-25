import cv2
from list_transfer_interface import load_model, list_style_transfer

# You can put content and style images in the corresponding path 
# and run this script to enjoy the effect of style transfer.

img_path = './content.jpg'
style_path = './styles'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

vgg, decoder = load_model()
output = list_style_transfer(img, style_path, vgg, decoder)
output_BGR = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
cv2.imwrite('transferred.jpg', output_BGR)

