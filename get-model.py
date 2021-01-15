'''get-model.shをpythonスクリプトに変換'''

# Exit on any error
# set -e
import sys, json, requests
from os import mkdir, chdir, path

# function usage()
def usage():
    msg = """\
Usage:
  $0 model-name

Options:
  model-name  Specify a model name to download. (e.g. bodypix/resnet50/float/model-stride16)
               Refer to https://storage.googleapis.com/tfjs-models for the available models.
"""
    print(msg)

# Check args
if len(sys.argv) < 2:
    usage()
    exit

# # Define constants & variables
BASE_URL   = "https://storage.googleapis.com/tfjs-models/savedmodel"
MODEL_NAME = sys.argv[1]
DIR_NAME   = MODEL_NAME.replace("/", "_")
URL  = BASE_URL+"/"+MODEL_NAME+".json"

# # Verify jq is installed
# [ -z ${JQ} ] && echo 'Please install "jq".' && exit 1

# # Fetch model.json and weights.bin
# mkdir ${DIR_NAME}
if not path.exists(DIR_NAME):
    mkdir(DIR_NAME)
chdir(DIR_NAME)

# wget -c -nv ${BASE_URL}/${MODEL_NAME}.json -O model.json
res = requests.get(URL)
if not res.ok:
    exit
lines = res.text
with open("model.json", "w") as f:
    f.write(lines)

# cat model.json |
#   ${JQ} -r ".weightsManifest | map(.paths) | flatten | @csv" |
#   tr "," "\n" |
#   xargs -I% wget -c ${BASE_URL}/${MODEL_NAME%/*}/%
data = json.loads(lines)
paths = [i["paths"] for i in data["weightsManifest"]]
paths = sum(paths, []) # flatten
pURL = URL[:URL.rindex("/") + 1]
for p in paths:
    res = requests.get(pURL + p)
    if not res.ok:
        continue
    with open(p, "wb") as f:
        f.write(res.content)
chdir("..")

print("Successfully downloaded to: {}".format(DIR_NAME))
