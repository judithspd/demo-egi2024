# Demo: Secure personalized federated learning within the AI4EOSC platform

_Authors: Judith Sáinz-Pardo and Álvaro López García (Instituto de Física de Cantabria, IFCA-CSIC-UC)._

**GOAL:** classify chest X-Ray images according to whether or not the patient has pneumonia. 
Original dataset extracted from [this study](https://www.sciencedirect.com/science/article/pii/S0092867418301545?via%3Dihub).

We divide the initial train data into 3 clients and we perform a stratified train-test random split: 75% train, 25% test.

The _adapFL_ strategy is implemented for performing the personalized step.

**Join the training:** client 4 has been created using the original validation set. **Try to connnect this client to the federated training!**.

Step by step guide: 

1. Clone this repository:

```bash
git clone https://github.com/judithspd/demo-egi2024.git
cd demo-egi2024
```

2. Secondly, you have to update the code for your client (Client 4), including the token ([line 24](https://github.com/judithspd/demo-egi2024/blob/main/demo_client4.py#L24)) and the UUID of the server ([line 86](https://github.com/judithspd/demo-egi2024/blob/main/demo_client4.py#L86)).

3. Then, create a virtual environment, install the requirements and run the code for your client:
   
```bash
virtualenv .venv -p python3
source .venv/bin/activate
pip install -r requirements.txt
python3 demo_client4.py
```

### References: 
- Data: Kermany, Daniel S., et al. "Identifying medical diagnoses and treatable diseases by image-based deep learning." cell 172.5 (2018): 1122-1131. [https://www.sciencedirect.com/science/article/pii/S0092867418301545?via%3Dihub](https://www.sciencedirect.com/science/article/pii/S0092867418301545?via%3Dihub)
- Flower python library: Beutel, Daniel J., et al. "Flower: A friendly federated learning research framework." arXiv preprint arXiv:2007.14390 (2020).
- Orginal study in the application of FL to the given data in this repository: Sáinz-Pardo Díaz, Judith, and Álvaro López García. "Study of the performance and scalability of federated learning for medical imaging with intermittent clients." Neurocomputing 518 (2023): 142-154. [https://www.sciencedirect.com/science/article/pii/S0925231222013844](https://www.sciencedirect.com/science/article/pii/S0925231222013844)
- Orignal study on the application of _adapFL_: Sáinz-Pardo Díaz, Judith, et al. "Personalized federated learning for improving radar based precipitation nowcasting on heterogeneous areas." Earth Science Informatics (2024): 1-24. [https://link.springer.com/article/10.1007/s12145-024-01438-9](https://link.springer.com/article/10.1007/s12145-024-01438-9).

### Funding and acknowledgments
This work is funded by European Union through the AI4EOSC project (Horizon Europe) under Grant number [101058593](https://ai4eosc.eu).
<p>
<img align="center" width="250" src="https://github.com/AI4EOSC/.github/raw/ai4eosc/profile/EN-Funded.jpg">
<img align="center" width="250" src="https://ai4eosc.eu/wp-content/uploads/sites/10/2023/01/horizontal-bg-white.png">
<p>

