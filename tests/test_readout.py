import json
import numpy as np
import pytest
from cns_tinker.readout import RidgeReadout


def test_fit_generalizes_and_roundtrips(tmp_path):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(120, 4)); y = x @ np.array([[1.], [-2.], [.5], [0.]]) + 3
    original = x.copy()
    model = RidgeReadout.fit(x[:80], y[:80], feature_names=["a","b","c","d"], output_names=["steer"], alpha=.001)
    assert model.evaluate(x[80:],y[80:],feature_names=model.feature_names)["steer"] < 1e-6
    np.testing.assert_array_equal(x,original)
    model.save(tmp_path/"model.json")
    restored = RidgeReadout.load(tmp_path/"model.json")
    np.testing.assert_array_equal(model.predict(x,feature_names=model.feature_names),restored.predict(x,feature_names=model.feature_names))
    with pytest.raises(FileExistsError): model.save(tmp_path/"model.json")
    with pytest.raises(ValueError): model.predict(x,feature_names=["b","a","c","d"])


def test_constant_features_dual_solver_and_bad_inputs(tmp_path):
    x=np.ones((3,8)); y=np.full((3,1),4.)
    names=[str(i) for i in range(8)]
    model=RidgeReadout.fit(x,y,feature_names=names,output_names=["value"])
    np.testing.assert_allclose(model.predict(x,feature_names=names),y)
    for alpha in [0,-1,float("nan")]:
        with pytest.raises(ValueError): RidgeReadout.fit(x,y,feature_names=names,output_names=["value"],alpha=alpha)
    with pytest.raises(ValueError): model.predict(np.full_like(x,np.nan),feature_names=names)
    model.save(tmp_path/"bad.json")
    data=json.loads((tmp_path/"bad.json").read_text());data["scale"][0]=0
    (tmp_path/"bad.json").write_text(json.dumps(data))
    with pytest.raises(ValueError): RidgeReadout.load(tmp_path/"bad.json")
