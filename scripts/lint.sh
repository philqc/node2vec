#!/bin/sh

mypy --ignore-missing-imports . || exit 1
flake8 || exit 1