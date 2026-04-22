#!/usr/bin/env sh


# Use addlicense (https://github.com/google/addlicense) to add Apache headers.

addlicense -c "Authors" -s -ignore '**/node_modules/**' -ignore '**/Osiris/**' .
