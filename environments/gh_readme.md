# Modelación de invernadero

## Variables

### Variables de estado

Para la modelación del invernadero se consideran 3 variables de estado: $T_{inv}$, $T_{ss}$ y $HR_{ext}$

* $T_{inv} [°C]$ : Temperatura al interior del invernadero
* $T_{ss} [°C]$: Temperatura en el subsuelo del invernadero.
* $X_{inv} [\frac{g}{m^3}$$]$: Humedad al interior del invernadero

### Variable manipulada

La variable manipulada corresponde a la apertura de la venta del invernadero $W[\%]$

### Perturbaciones

Como perturbaciones se consideran

* $I_s [\frac{W}{m^2}]$: Radiación solar
* $T_{ext} [°C]$: Temepratura exterior
* $HR_{ext} [\%]$: Humedad relativa exterior

## Dinámica

### Dinámica de la humedad

La tasa de cambio de la humedad viene dada por la ecuación

$$
\frac{dX_{inv}}{dt} = \frac{1}{h}\cdot(E-C-V)
$$

Donde $E [\frac{g }{ m^2 \cdot s}] $ corresponde al flujo de vapor debido a la transpiración de la planta, $C [\frac{g }{ m^2 \cdot s}]$ es el flujo de humedad concerniente a la condensación y $V [\frac{g }{ m^2 \cdot s}]$ es el flujo de humedad concerniente a la ventilación.

#### Evapotranspiración

$$
E=g_E\left(x_{\text {crop }}-x_{inv}\right)
$$

Donde $g_E$ corresponde a la transmisividad y se define como

$$
g_E=\frac{2 L A I}{(1+0.7584 \exp \left(0.0518 T_{inv}\right)) r_b+r_s}
$$

Siendo los términos $r_b$ y $r_s$

$$
r_b=\frac{1,174 d^{0.5}}{\left(d\left|T_{\text {cu}}-T_{inv}\right|+207 \varphi_{\text {wind }}^2\right)^{0.25}}
$$

$$
r_s=82\left(\frac{4.3}{0.54}\right)\left[\exp \left(\frac{-k R_n}{2 L A I}\right)\right]\left(1+0.023\left(T_{inv}-20\right)^2\right)
$$

$$
R_n=\tau(1-\exp (-0.7 L A I)) I_s
$$

$$
x_{crop} = 5.5638 \exp(0.0572 \cdot T_{inv}) + \varepsilon \frac{r_b}{2 L A I} \frac{R_n}{\lambda}
$$

$$
HR_{inv}=\frac{100 x_{inv}}{5.5638 \exp \left(0.0572 \cdot T_{inv}\right)}
$$

#### Condensación

$$
C = g_c\left[0.2522 \exp \left(0.0485 \cdot T_{inv}\right)\left(T_{inv}-T_{ext}\right)-\left(x_{inv}^*-x_{inv}\right)\right]
$$

$$
g_c \cong \max \left[0,0.25 \cdot 10^3\left(T_{inv}-T_{\text {cu }}\right)^{1 / 3}\right]
$$

#### Ventilación

$$
V  [\frac{g }{ m^2 \cdot s}]= g_v [\frac{m}{s}] \cdot (X_{inv} - X_{ext})[\frac{g}{m^3}]
$$

$$
g_v [m^3/{s}]= (\varphi_{lek}+\varphi_{win})
$$

$$
\varphi_{lek} = 0.000083 + 0.000035 V_0 f_a
$$

$$
\varphi_{win} \approx |A_{window} [\%] \cdot w_{wind} [m/s]|
$$

### Dinámica de la temperatura del aire

$$
\frac{d}{dt} T_{inv} = \frac{Q_{rad} - Q_{ss} - Q_{cc} - Q_{evp} - Q_{ren}}{(\rho_{air} \cdot c_{pa} + x_{inv} \cdot c_{pv}) \cdot V_{inv}}
$$

$$
Q_{rad} = S_s \cdot [I_s \cdot (\alpha + \tau \alpha_s)] + S_c \cdot \sigma \cdot \tau_{ter} \cdot (\epsilon_{atm} \cdot T_{atm}^4 - \epsilon_{ter} \cdot T_c^{4})
$$

$$
Q_{evp} = \lambda_0 \cdot ET_c
$$

$$
Q_{ss} = K_s \cdot S_c \cdot\left(T_{inv}-T_{ss}\right) / p
$$

$$
Q_{\text {ren}}={V}_{\text {inv }} \cdot  R \cdot \rho \cdot\left[{c}_{{pa}} \cdot\left(T_{inv} - T_{ext}\right)+\lambda_0 \cdot\left(x_{i}-x_{e}\right)\right. \left.+ c_{pv} \cdot\left(x_{i} \cdot T_{inv}-x_{e} \cdot T_{ext}\right)\right]
$$

$$
Q_{cc} = S_d \cdot K_{cc} \cdot (T_{inv} - T_{ext})
$$

### Dinámica de la temperatura del suelo
