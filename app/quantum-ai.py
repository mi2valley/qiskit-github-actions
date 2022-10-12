import os
from qiskit import IBMQ
from qiskit import QuantumCircuit, execute, ClassicalRegister, QuantumRegister
from qiskit.visualization import *
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt
import io
import warnings
from IPython.display import display
from PIL import Image
# from stability_sdk import client
# import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import deepl


IBMQ_TOKEN = os.getenv("IBMQ_TOKEN")
IBMQ.save_account(str(IBMQ_TOKEN))

# STABILITY_KEY = os.getenv("STABILITY_KEY")
# stability_api = client.StabilityInference(
#     key=str(STABILITY_KEY), 
#     verbose=True,
# )


# アカウント情報をロードして、使える量子デバイスを確認します
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
provider.backends()

# 最もすいているバックエンドを選びます
large_enough_devices = IBMQ.get_provider(hub='ibm-q', group='open', project='main').backends(
    filters=lambda x: x.configuration().n_qubits > 4 and not x.configuration().simulator and x.configuration().quantum_volume > 16 ) # and not x.configuration().backend_name == "ibmq_manila" and not x.configuration().backend_name == "ibmq_bogota")
print(large_enough_devices)
real_backend = least_busy(large_enough_devices)

print("ベストなバックエンドは " + real_backend.name())

# build quantum circuit
qreg_q = QuantumRegister(5, 'q')
creg_c = ClassicalRegister(5, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.rz(1.5707963267948966, qreg_q[0])
circuit.sx(qreg_q[0])
circuit.rz(1.5707963267948966, qreg_q[0])
circuit.rz(1.5707963267948966, qreg_q[1])
circuit.sx(qreg_q[1])
circuit.rz(1.5707963267948966, qreg_q[1])
circuit.rz(1.5707963267948966, qreg_q[2])
circuit.sx(qreg_q[2])
circuit.rz(1.5707963267948966, qreg_q[2])
circuit.rz(1.5707963267948966, qreg_q[3])
circuit.sx(qreg_q[3])
circuit.rz(1.5707963267948966, qreg_q[3])
circuit.rz(1.5707963267948966, qreg_q[4])
circuit.sx(qreg_q[4])
circuit.rz(1.5707963267948966, qreg_q[4])
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.measure(qreg_q[2], creg_c[2])
circuit.measure(qreg_q[3], creg_c[3])
circuit.measure(qreg_q[4], creg_c[4])

# execute
# job = execute(circuit, real_backend)
# job_id = job.job_id()
# print(job_monitor(job))
# job_result = job.result()
# job_counts = job_result.get_counts()
# print(job_counts)
# plot_histogram(job_counts)

# seedAs = open('./seedAs.json', 'r')
# seedAs_dict = json.load(seedAs)
# print('json_dict:{}'.format(type(seedAs_dict))) #dict

# seedBs = open('./seedBs.json', 'r')
# seedBs_dict = json.load(seedBs)
# print('json_dict:{}'.format(type(seedBs_dict))) #dict

# seedCs = open('./seedCs.json', 'r')
# seedCs_dict = json.load(seedCs)
# print('json_dict:{}'.format(type(seedCs_dict))) #dict


# answers = stability_api.generate(
#     prompt="a photograph of an astronaut riding a horse"
# )

# for resp in answers:
#     for artifact in resp.artifacts:
#         if artifact.finish_reason == generation.FILTER:
#             warnings.warn(
#                 "Your request activated the API's safety filters and could not be processed."
#                 "Please modify the prompt and try again.")
#         if artifact.type == generation.ARTIFACT_IMAGE:
#             img = Image.open(io.BytesIO(artifact.binary))
#             display(img)

# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_type=torch.float16, revision="fp16")
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  

