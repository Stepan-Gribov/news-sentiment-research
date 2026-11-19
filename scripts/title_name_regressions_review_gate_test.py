"""Independent review gating must happen before fitted signal/outcome slopes."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

from title_name_regressions_core import IntegrityError,sha256_file
import run_title_name_regressions as runner


class ReviewGateChecks(unittest.TestCase):
    def audit(self):return dict(audit_execution_complete=True,status="AUDIT_COMPLETE",llm=dict(status="READY",integrity_failures=[]))

    def test_absent_review_stops_before_signal_or_outcome_estimation(self):
        with tempfile.TemporaryDirectory(prefix="r4_gate_") as temp,patch.object(runner,"announce"),patch.object(runner,"fit_signal") as signal,patch.object(runner,"fit_panel_fe") as outcome:
            result=runner.estimate({},self.audit(),Path(temp),dict(w2_input=dict(version="v1")),Path("unused"),Path("unused"))
            self.assertIsNone(result);signal.assert_not_called();outcome.assert_not_called()

    def test_scope_not_approved_never_fits_coefficients(self):
        with tempfile.TemporaryDirectory(prefix="r4_gate_") as temp,patch.object(runner,"announce"),patch.object(runner,"fit_signal") as signal:
            root=Path(temp);(root/"config").mkdir()
            (root/"config/crosswalk_review_gate.json").write_text(json.dumps(dict(allow_exploratory_DW=False)),encoding="utf-8")
            self.assertIsNone(runner.estimate({},self.audit(),root,dict(w2_input=dict(version="v1")),Path("unused"),Path("unused")))
            signal.assert_not_called()

    def test_changed_crosswalk_or_review_bytes_reject_before_any_fit(self):
        with tempfile.TemporaryDirectory(prefix="r4_gate_") as temp,patch.object(runner,"announce"),patch.object(runner,"fit_signal") as signal:
            root=Path(temp);(root/"config").mkdir();spec=root/"config/spec.json";spec.write_text("{}",encoding="utf-8")
            review=root/"config/review.json";review.write_text("{}",encoding="utf-8")
            gate=dict(allow_exploratory_DW=True,scope="EXPLORATORY_DW_ONLY",w2_version="other",spec_sha256=sha256_file(spec),
                      review=dict(path=str(review),sha256="0"*64))
            path=root/"config/crosswalk_review_gate.json";path.write_text(json.dumps(gate),encoding="utf-8")
            with self.assertRaises(IntegrityError):runner.estimate({},self.audit(),root,dict(w2_input=dict(version="v1")),spec,Path("unused"))
            gate["w2_version"]="v1";path.write_text(json.dumps(gate),encoding="utf-8")
            with self.assertRaises(IntegrityError):runner.estimate({},self.audit(),root,dict(w2_input=dict(version="v1")),spec,Path("unused"))
            signal.assert_not_called()


if __name__=="__main__":unittest.main(verbosity=2)
