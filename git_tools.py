import json
import os.path
import shutil
import subprocess
import sys
import logging
import platform as plat

DOCKER_IMAGE_LINGUIST = "linguist"
DOCKER_IMAGE_LICENSEE = "licensee"
TOOLS = {
    "linguist": DOCKER_IMAGE_LINGUIST,
    "licensee": DOCKER_IMAGE_LICENSEE
}

# Set up logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def check_docker():
    """Check if Docker is installed and running."""
    try:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.PIPE)
        logging.info("Docker is installed and running")
        return True
    except subprocess.CalledProcessError:
        logging.error("Docker is not installed or not running")
        raise


def build_linguist_image():
    """Build the Docker image for GitHub-Linguist."""
    logging.info("Cloning Github-Linguist and building docker image")
    # Check if the Docker image already exists
    result = subprocess.run(["docker", "images", "-q", DOCKER_IMAGE_LINGUIST], capture_output=True, text=True)
    if result.stdout.strip():
        logging.info("Github-Linguist image already exists.")
    else:
        if not os.path.isdir("linguist"):
            subprocess.run(["git", "clone", "https://github.com/github-linguist/linguist"], check=True)
        subprocess.run(["docker", "build", "-t", DOCKER_IMAGE_LINGUIST, "linguist"], check=True)
        shutil.rmtree("linguist")
        logging.info("Github-Linguist image built successfully.")


def build_licensee_image():
    """Clone and build the Docker image for GitHub-Licensee."""
    logging.info("Cloning Github-Licensee and building docker image")
    # Check if the Docker image already exists
    result = subprocess.run(["docker", "images", "-q", DOCKER_IMAGE_LICENSEE], capture_output=True, text=True)
    if result.stdout.strip():
        logging.info("Licensee image already exists.")
    else:
        if not os.path.isdir("licensee"):
            subprocess.run(["git", "clone", "https://github.com/licensee/licensee"], check=True)
        subprocess.run(["docker", "build", "-t", DOCKER_IMAGE_LICENSEE, "licensee"], check=True)
        shutil.rmtree("licensee")
        logging.info("Licensee image built successfully.")


def analyze_directory(tool, path):
    """Analyze a directory using the specified GitHub tool inside the Docker container."""
    # IF THERE IS AN ISSUE FOR LINUX PLEASE ADD :Z AFTER THE DOCKER PATH
    if plat.system() == "Windows":
        docker_path = path[2:]
    else:
        docker_path = path
    if tool == "linguist":
        result = subprocess.run([
            "docker", "run", "--rm", "-v", f"{path}:{docker_path}:Z", "-w", docker_path, "-t", DOCKER_IMAGE_LINGUIST, "github-linguist", "--json"
        ], check=True,capture_output=True,text=True)
    else:
        result = subprocess.run([
            "docker", "run", "--rm", "-v", f"{path}:{docker_path}", DOCKER_IMAGE_LICENSEE, "detect", docker_path, "--json"
        ], check=True,capture_output=True,text=True)
    return json.loads(result.stdout)
