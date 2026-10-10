# Deploying a Gradio app with Docker

Tags: DEPLOYMENT, DOCKER


### Introduction

Gradio is a powerful and intuitive Python library designed for creating web apps that showcase machine learning models. These web apps can be run locally, or [deployed on Hugging Face Spaces ](https://huggingface.co/spaces)for free. Or, you can deploy them on your servers in Docker containers. Dockerizing Gradio apps offers several benefits:

- **Consistency**: Docker ensures that your Gradio app runs the same way, irrespective of where it is deployed, by packaging the application and its environment together.
- **Portability**: Containers can be easily moved across different systems or cloud environments.
- **Scalability**: Docker works well with orchestration systems like Kubernetes, allowing your app to scale up or down based on demand.

## How to Dockerize a Gradio App

Let's go through a simple example to understand how to containerize a Gradio app using Docker.

#### Step 1: Create Your Gradio App

First, we need a simple Gradio app. Let's create a Python file named `app.py` with the following content:

```python
import gradio as gr

def greet(name):
    return f"Hello {name}!"

iface = gr.Interface(fn=greet, inputs="text", outputs="text").launch()
```

This app creates a simple interface that greets the user by name.

#### Step 2: Create a Dockerfile

Next, we'll create a Dockerfile to specify how our app should be built and run in a Docker container. Create a file named `Dockerfile` in the same directory as your app with the following content:

```dockerfile
FROM python:3.10-slim

WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir gradio
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
```

This Dockerfile performs the following steps:
- Starts from a Python 3.10 slim image.
- Sets the working directory and copies the app into the container.
- Installs Gradio (you should install all other requirements as well).
- Exposes port 7860 (Gradio's default port).
- Sets the `GRADIO_SERVER_NAME` environment variable to ensure Gradio listens on all network interfaces.
- Specifies the command to run the app.

#### Step 3: Build and Run Your Docker Container

With the Dockerfile in place, you can build and run your container:

```bash
docker build -t gradio-app .
docker run -p 7860:7860 gradio-app
```

Your Gradio app should now be accessible at `http://localhost:7860`.

## Important Considerations

When running Gradio applications in Docker, there are a few important things to keep in mind:

#### Running the Gradio app on `"0.0.0.0"` and exposing port 7860

In the Docker environment, setting `GRADIO_SERVER_NAME="0.0.0.0"` as an environment variable (or directly in your Gradio app's `launch()` function) is crucial for allowing connections from outside the container. And the `EXPOSE 7860` directive in the Dockerfile tells Docker to expose Gradio's default port on the container to enable external access to the Gradio app. 

#### Running Multiple Replicas

A Gradio app keeps a session's state, files, `auth=`, and `@gr.render` state in one process. With multiple replicas behind a load balancer that has no session affinity, a request can reach a replica that never saw the session, which produces a `session_not_found` error or silently empty state. There are two ways to run multiple replicas:

1. **Enable session affinity (default mode).** Turn on stickiness with `sessionAffinity: ClientIP` so all requests from the same user reach the same instance. This is required because Gradio's communication protocol needs multiple connections from the frontend to reach the same backend for an event to be processed correctly. (If you use Terraform, add a [stickiness block](https://registry.terraform.io/providers/hashicorp/aws/3.14.1/docs/resources/lb_target_group#stickiness) to your target group definition.) Affinity still fails when an autoscaler removes the instance holding a session.

2. **Enable multi-replica mode (no affinity required).** Opt in with `launch(multi_replica=...)` (or the `GRADIO_MULTI_REPLICA` environment variable) to store session state and files in shared backends, so any replica can serve any request. This removes the need for `sessionAffinity`/sticky cookies:

   ```python
   demo.launch(
       multi_replica={
           "session": {"url": os.environ["GRADIO_REDIS_URL"]},
           "files": {"bucket": "your-org/your-bucket", "token": os.environ["HF_TOKEN"]},
           "auth_secret": os.environ["GRADIO_AUTH_SECRET"],
           "queue": {"url": os.environ["GRADIO_REDIS_URL"], "lease_ms": 60000},
           "drain_window": 20,
       }
   )
   ```

   Multi-replica mode requires:
   - **A Redis backend** for session state and the durable work queue, supplied by the operator.
   - **A Hugging Face Storage Bucket** (S3-compatible) for uploaded and generated files.
   - **`drain_window` less than the queue's `lease_ms`**, so an interrupted job is redelivered after the replica has stopped. Launch fails if this is violated.
   - **Serializable session state.** Only values the store's typed envelope can represent cross a replica: JSON-native types, plus bytes, `datetime`/`date`/`time`, `Decimal`, `set`, `tuple`, and string/integer dict keys. A value outside this set fails with an error naming the component and its type. Gradio does not fall back to affinity. See the serialization requirement below.

#### Serialization requirement (multi-replica mode)

With an external store configured, a `gr.State` value that cannot be represented by the store's typed envelope is rejected when it is written, with an error naming the state and the value's type. Values that previously survived via silent stringification (for example, arbitrary objects) are no longer accepted. Convert such state to a supported representation (for example, a dict of primitives) before enabling multi-replica mode.

Request-scoped component values do not cross the store; only `gr.State` values do. Component configuration is rebuilt from each replica's own app.

#### Trust boundary and lifecycle (multi-replica mode)

The external store holds every tenant's session state and any secrets kept in `gr.State`, so treat it as a security boundary:

- **Credentials** are operator-supplied and validated at launch; the Redis client is imported only when multi-replica mode is on, so the default install has no extra dependency.
- **Keys are namespaced** per app and tenant.
- **Sessions are authorized**, not merely keyed: a session hash that belongs to another principal is a not-found, not a readable session.
- **Retention:** a closed session is retained for its TTL and then removed, and files orphaned by an expired session are collected.
- **No `auth=`?** Session and file isolation is limited to the client session identity. Without `auth=`, there is no cross-principal isolation beyond that, so do not rely on multi-replica mode to separate untrusted users.

#### Deploying Behind a Proxy

If you're deploying your Gradio app behind a proxy, like Nginx, it's essential to configure the proxy correctly. Gradio provides a [Guide that walks through the necessary steps](https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx). This setup ensures your app is accessible and performs well in production environments.

