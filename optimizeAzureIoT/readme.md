## Introduction

Over the last few years IoT devices and ML/AI have become very popular, and now a lot of companies are moving forward to use them in production. All cloud providers, including Microsoft Azure, provides services how to deploy developed machine learning algorithms to the edge device. The main concern of some industries (automotive, agriculture, etc.) is that in production the cost for data transfer, out of the total cost of ownership, will be huge.

First of all, let's take a look at how Azure ML IoT works and when reducing the data transfer matters.

There are multiple situations when it's needed to work with prediction in "offline" mode (when the device doesn't have the direct access to the internet). Here are a few of them:

* Underground facilities (parking lots, basement, some part of factories without a WiFi connection)
* Movables devices (vehicles, planes, drones, etc.)

## Azure ML IoT general overview
Before we continue let's take a look at two different types of devices and how they could be connected to the cloud.

<img src="images/Different use-cases.png" />

As we can see usually non-movable IoT devices (for instance, factories) use WiFi or Ethernet to connect to the internet. Other types of devices are movable and for some industries (e.g. automotive, maps development, agriculture) mobile network is the only one available type of the connection for them (e.g. drones, vehicles).

Azure does not differentiate two types of devices (i.e. static and movable) and provides a single approach for them. The diagram below illustrates Azure IoT ML workflow on a high level:

<img src="images/General ML.png" />

The main stages here are:

1.  IoT device sends data to the Azure Event Hub. This is raw data transfer (usually via MQTT or HTTP connection). Be careful, Azure SDK is not very flexible currently and does not provide the ability to downsample the data, a developer needs to implement downsampling process.
2. Azure provides a really easy way to set up the whole pipeline to move data from the IoT device to the data lake. You could take a look at the proposed architecture from the Microsoft team on how to move data from IoT devices to the cloud [here](https://docs.microsoft.com/en-us/azure/architecture/example-scenario/data/realtime-analytics-vehicle-iot).
3. When data is landed in the Azure Data Lake Storage gen2 an engineer can start the development of their own algorithms. If you are a developer, you could start your journey with Azure ML and IoT [here](https://docs.microsoft.com/en-us/azure/iot-edge/tutorial-machine-learning-edge-01-intro).
4. After the development of the algorithm has been done, an engineer needs to create a Docker container with the serialized model (standard serialization for Azure ML is Python [pickle](https://docs.python.org/3/library/pickle.html) library).
5. After that, Azure IoT Hub transfers the newly generated Docker container to the IoT device.
6. And the last step - update the IoT device with a newly generated Docker container. IoT Edge core provides the ability to monitor and detect automatically when a new image is available for the IoT device and start the update process automatically, which is amazing. (Azure uses [moby project](https://mobyproject.org/) to do a lot of things under the hood). Azure moved from Docker CE/EE to moby recently (more information about containers engines you could find [here](https://docs.microsoft.com/en-us/azure/iot-edge/support)).

What could go wrong here? Usually, it is needed to update the model at least once per month (steps #4 - #6). The amount of data, which need to be transferred from cloud to the device is big, but not critical for 1 device **(60Mb per model)**. Although the update for 1,000 devices will be 60,000 Mb (60Gb) at a time. 1Gb of shared data (for 500 sim cards) for AT&T business contract in the US costs [720\$](https://marketplace.att.com/products/iot-share-plan-lte-na). **This means that 1 update for 1,000 devices costs 1,500\$.** Companies like deliveries usually have about 100,000 vehicles so the estimated price for them is approximately **150,000\$ per month**.

Is it possible to reduce the 60Mb per model?

## Azure ML IoT Docker container deep dive

Microsoft team is doing a great job in writing the documentation (especially tutorials) for all of the services. Microsoft Azure team provides the following guide on how to [deploy Azure Machine Learning as an IoT Edge module](https://docs.microsoft.com/en-us/azure/iot-edge/tutorial-deploy-machine-learning).

Following this tutorial it is possible to develop own anomaly detection algorithm and deploy it to the IoT device.

One of the first actions which you need to do - get python notebook from their GitHub [repository](https://github.com/Azure/ai-toolkit-iot-edge). Let's take a closer look at how they developed the ability to create a Docker container with a pickled model in it (Part 4.2 Create Docker Image in the notebook).

```
from azureml.core.image import Image, ContainerImage


image_config = ContainerImage.image_configuration(runtime= "python",
                                 execution_script="iot_score.py",
                                 conda_file="myenv.yml",
                                 tags = {'area': "iot", 'type': "classification"},
                                 description = "IOT Edge anomaly detection demo")




image = Image.create(name = "tempanomalydetection",
                     # this is the model object 
                     models = [model],
                     image_config = image_config, 
                     workspace = ws)

image.wait_for_creation(show_output = True)
```

As you can see they are triggering the function `create()` from azure.core.image package. Unfortunately, I can't find the source code for it on GitHub and will appreciate it if someone could point me out to it in comments.

During each run of that command Azure Python notebooks will store the whole log in Azure Storage. The log file is stored in the new storage, which you can find in the Storage Explorer (usually name of your project + random alphanumeric sequence) the blob container name is `azureml` and the folder - `ImageLogs`. Inside it there is a set of folders for each `Image.create()` run. You could find my build.log file [here](https://gist.github.com/gron1987/837c98175f2252ffe393866cb464eb15).

How the docker image creation process looks like (the first command is on bottom)?

<img src="images/Docker layers.png" />

If you want to deep dive what is unicorn, nginx, flask, etc. I recommend you to take a look at Paige Liu's blog post ["Inside the Docker image built by Azure Machine Learning service"](https://liupeirong.github.io/amlDockerImage/).

What is interesting here - Microsoft team placed a newly generated model (model.pkl) file on stage #5. **The model size itself is only 6Kb.** But the docker image layers diff size is 60Mb (I've checked that on the device, 60Mb was transferred from the cloud to the device).

During the docker creation process in our python notebook we have the following code:

```
# This specifies the dependencies to include in the environment
from azureml.core.conda_dependencies import CondaDependencies 


myenv = CondaDependencies.create(conda_packages=['pandas', 'scikit-learn', 'numpy'])


with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())
```

Microsoft provides the ability to select which conda packages are required to be installed on the device, which is great. But on which layer they are deploying it in the docker container? As we can see from the layered images above - on layer #11. What is the size of this layer?

<img src="images/conda packages size.png" />

60Mb as an archive (you can find the size of the layer in the meta-information for your container in the Azure Container registry). If you are not familiar with Docker images I should explain it a little bit more here why this is important and why this layer on "top" means we need to transfer it all the time to the edge device.

## Docker image layers overview

Each Docker container contains a base image and then a number of layers (unlimited) with additional files in it. Each layer (including base image layer) has its sha5 hash, which is almost unique. The image below shows how this works (Thank you [cizxs](https://stackoverflow.com/users/1925083/cizixs) for this diagram, since it's not available now on Docker official web page)

<img src="https://media-exp1.licdn.com/dms/image/C4D12AQFKyx9FHmgufg/article-inline_image-shrink_1500_2232/0?e=1596067200&v=beta&t=4MUFBXeaxCdHKKQqjkcf6PpmbqHez1dx37D4NjR4Zto" />

During the "pull" docker checking in the local cache for that sha5 number and if a layer already exists then there's no need to download it from the server. This reduces the size, which we need to transfer between Docker repository and end device. Usually the docker size for python with all DS libraries is ~1Gb, but with this layered approach we need to transfer only a small amount of this information after the first setup (you can find more information on the internet, but I recommend to start from [this Stackoverflow answer](https://stackoverflow.com/questions/31222377/what-are-docker-image-layers/51660942#51660942)).

Each time we run Docker command (RUN / COPY / etc.), we are building a new layer. This newly generated layer will have its cryptographic hash. For us it means that each new run of "Images.create()" function will generate a new layer with conda packages, even if we had not modified that section.

## Azure ML IoT Docker container optimizations

Without any changes Docker container layers look like that:

<img src="images/Image diff.png" />

As you can see we've discovered where our 60Mb came from. But what we could do with that? In theory, there are multiple steps which we could try:
* Update the Docker file and avoid any dependencies (you could do your base image in theory). Microsoft provides you the instruction on how to do that [https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-custom-docker-image](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-custom-docker-image).
* Modify python notebook.

Solution #1 will not work because during steps #6-#11 Docker image also installs a lot of other components including Azure services, there's no ability to override them.

Those services are already available after the first installation and already available on the edge device, so could we try to re-use them from 1st image instead of trying to transfer them all the time?

First of all, we need to create a Docker image which will be based on the 1st version of the image, which is already on the device.

```
#Path to the image without tags
base_image_path = image.image_location[:image.image_location.rfind(":")]
#Registry path only
registry = image.image_location.split(".")[0]
#New tag version
version_new=image.version

#Dockerfile text
dockerfile = """
FROM {base_image_path}:{version_new} AS model
FROM {base_image_path}:{version_old}


COPY --from=model /var/azureml-app/azureml-models /var/azureml-app/azureml-models
COPY --from=model /var/azureml-app/iot_score.py /var/azureml-app/iot_score.py
""".format(base_image_path=base_image_path,version_new=image.version,version_old=1).strip()

#Store as lock Dockerfile file
%store dockerfile > Dockerfile

#Run new "build" stage for the newly generated Dockerfile via Azure Container Registry

!az acr build --image $image.name:iot-$version_new --registry $registry --file Dockerfile .
```

This code snippet shows how to copy azureml-models directory (by default this is a directory for model.pkl files) and iot_score.py (file to be executed on the edge device) from newly generated Docker image (with new layers) to the old version of Docker image (to avoid transferring conda dependencies). This is suitable only if conda dependencies list was not modified. The updated image will be stored in the same repository but with tag "iot-{version_new}", where version new is a new tag, which was generated automatically for this image (auto-incremental number).

You should put this script right after you test your image but before chapter 6 (Deploy container to Azure IoT Edge device) (or at least as the first step in it).

Below you could find how that impact the layers:

<img src="images/Image diff after optimization.png" />

As you can see we've updated just 2 layers (you could do two COPY commands in one to have only 1 layer difference if you want).

The total size for these 2 layers is ~2Kb.

<img src="images/model packages size.png" />

We also need to change the deployment part:
```
# Update the workspace object
ws = Workspace.from_config()
image_location = image.image_location[:image.image_location.rfind(":")] + ":iot-" + str(image.version)


# Getting your container details
container_reg = ws.get_details()["containerRegistry"]
reg_name=container_reg.split("/")[-1]
container_url = "\"" + image_location + "\","
subscription_id = ws.subscription_id
print('{}'.format(image_location))
print('{}'.format(reg_name))
print('{}'.format(subscription_id))
from azure.mgmt.containerregistry import ContainerRegistryManagementClient
from azure.mgmt import containerregistry
client = ContainerRegistryManagementClient(ws._auth,subscription_id)
result= client.registries.list_credentials(resource_group_name, reg_name, custom_headers=None, raw=False)
username = result.username
password = result.passwords[0].value
```

And deployment.json configuration
```
file = open('iot-workshop-deployment-template.json')
contents = file.read()
contents = contents.replace('__MODULE_NAME', module_name)
contents = contents.replace('__REGISTRY_NAME', reg_name)
contents = contents.replace('__REGISTRY_USER_NAME', username)
contents = contents.replace('__REGISTRY_PASSWORD', password)
contents = contents.replace('__REGISTRY_IMAGE_LOCATION', image_location)
with open('./deployment.json', 'wt', encoding='utf-8') as output_file:
    output_file.write(contents)
```

## Conclusion
We've just reduced the size for the Docker image layers, which need to be transferred to the IoT device from 60Mb to 2Kb. Now the update of the model in production will cost you only a few cents.