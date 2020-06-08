#!/bin/bash

pip3 install jupyter_contrib_nbextensions jupyter_nbextensions_configurator

jupyter contrib nbextension install --user && \
jupyter nbextensions_configurator enable --user && \
jupyter nbextension enable splitcell/splitcell && \
jupyter nbextension enable codefolding/main && \
jupyter nbextension enable execute_time/ExecuteTime && \
jupyter nbextension enable varInspector/main && \
jupyter nbextension enable snippets_menu/main && \
jupyter nbextension enable toggle_all_line_numbers/main && \
jupyter nbextension enable latex_envs/latex_envs