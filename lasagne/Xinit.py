from . import nonlinearities

__all__ = [
    "GlorotUniformGain",
]

GlorotUniformGain = {};
GlorotUniformGain[nonlinearities.sigmoid] = 4.0;
GlorotUniformGain[nonlinearities.softmax] = 1.0;
GlorotUniformGain[nonlinearities.tanh] = 1.0;
GlorotUniformGain[nonlinearities.rectify] = 1.0;
GlorotUniformGain[nonlinearities.LeakyRectify] = 1.0;
GlorotUniformGain[nonlinearities.leaky_rectify] = 1.0;
GlorotUniformGain[nonlinearities.very_leaky_rectify] = 1.0;
#GlorotUniformGain[nonlinearities.ScaledTanH] = 1.0;
#GlorotUniformGain[nonlinearities.elu] = 1.0;
#GlorotUniformGain[nonlinearities.softplus] = 1.0;
GlorotUniformGain[nonlinearities.linear] = 1.0;
GlorotUniformGain[nonlinearities.identity] = 1.0;