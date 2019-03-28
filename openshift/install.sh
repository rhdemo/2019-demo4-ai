#!/usr/bin/bash

ORG="vpavlin"
REPO="2019-demo4-ai"


[ -z "$GIT_REF" ] && GIT_REF="master"
[ -n "$LOCAL" ] && GIT_REF=

function _install() {
    f=$1
    if [ -z "${LOCAL}" ]; then
        f="https://raw.githubusercontent.com/${ORG}/${REPO}/${GIT_REF}/openshift/$1?time=`date +%N`"
    fi
    echo "==> ${f}"
    oc apply -f ${f}
}

_install "jupyter-tensorflow.json"
_install "config.yaml"
_install "tf-serving.bc.yaml"
_install "serving.knative.yaml"

#istio needs this:
oc adm policy add-scc-to-user anyuid -z default
oc adm policy add-scc-to-user privileged -z default