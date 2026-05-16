# LoopGain developer Makefile.
#
# Targets here are for *manual* invocation by the maintainer. There is no
# CI integration for `examples` — running them costs real Anthropic API
# spend, so they should only fire when a human asks.

PYTHON ?= .venv/bin/python

EXAMPLES = $(wildcard examples/0?_*.py)

.PHONY: help test examples examples-haiku $(EXAMPLES)

help:
	@echo "Targets:"
	@echo "  test          Run the full pytest suite"
	@echo "  examples      Run all examples 01-07 sequentially (costs API spend)"
	@echo "  examples/0X_<name>.py  Run a single example"
	@echo "  examples-haiku  Run example 01 with claude-haiku-4-5"

test:
	$(PYTHON) -m pytest -q

# Each example runs baseline + LoopGain (dual run) and POSTs telemetry.
# Roughly $0.10-$0.30 per example with Opus; ~$1-3 for a full sweep.
examples: $(EXAMPLES)

$(EXAMPLES):
	@echo "→ $@"
	$(PYTHON) $@

examples-haiku:
	@echo "→ example 01_code_pytest with Haiku 4.5"
	LOOPGAIN_EXAMPLE_MODEL=claude-haiku-4-5-20251001 $(PYTHON) examples/01_code_pytest.py
