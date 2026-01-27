# Assignment 4  
## Curve Fitting and Interpolation Using Numerical Methods

---

## Part 1: Curve Fitting Using the Method of Least Squares

### Objective
The objective of this part is to apply the method of least squares to fit different regression models to given experimental data sets and to evaluate the goodness of fit using the Sum of Squared Errors (SSE) and the coefficient of determination (R²).

---

### Data Sets
Three data sets (A, B, and C) were provided. For each data set, five regression models were fitted:
- Linear model  
- Parabolic (second-order polynomial) model  
- Power model  
- Exponential model  
- Logarithmic model  

---

### Methodology
For each model, the parameters were estimated using the least squares method.  
The goodness of fit was evaluated using:

- **Sum of Squared Errors (SSE):**

\[
SSE = \sum (y_i - \hat{y}_i)^2
\]

- **Coefficient of Determination (R²):**

\[
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]

Graphs were plotted for each data set showing the original data points and the fitted curves.

---

### Results and Discussion
For each data set, the fitted models were compared numerically using SSE and R² and visually using plots.

The model with the **lowest SSE** and **highest R²** was considered the best fit.  
Visual inspection of the plots confirmed the numerical results.

Overall, different data sets favored different models, demonstrating the importance of selecting a regression model based on both numerical error analysis and visual behavior.

---

### Conclusion (Part 1)
The least squares method provides an effective way to approximate experimental data using regression models. The combination of SSE, R², and graphical analysis allows reliable comparison between models and selection of the most suitable one for a given data set.

---

## Part 2: Interpolation Using Numerical Methods

### Objective
The objective of this part is to construct interpolation functions for population data using Newton’s interpolation, Lagrange interpolation, and Cubic Spline interpolation, and to compare their behavior and predictions.

---

### Data Description
Population data for a city were given for the years 1990, 1995, 2000, 2005, 2010, and 2015.

---

### Methods Used
The following interpolation methods were implemented in Python:

- **Newton’s interpolation** using divided differences  
- **Lagrange interpolation**  
- **Cubic spline interpolation**

The interpolation curves were constructed to pass exactly through all given data points.

---

### Results
- A divided difference table was constructed for Newton’s method.
- Population values were estimated for each year from 1990 to 2015.
- Predictions were made for the years **2018** and **2025**.
- A scaling study was performed using:
  - years since 1990
  - population in millions

Graphs were plotted showing the data points and interpolation curves for all three methods.

---

### Discussion
Newton and Lagrange interpolation produce the same interpolating polynomial and therefore give identical or nearly identical results.  
The cubic spline interpolation produces a smoother curve and avoids strong oscillations that may occur with high-degree polynomials.

For extrapolation beyond the given data range (2018 and 2025), noticeable differences between methods were observed. This highlights the uncertainty of extrapolation and the importance of careful interpretation of predicted values.

Using scaled variables improves numerical stability and reduces rounding errors.

---

### Conclusion (Part 2)
Among the interpolation methods, cubic spline interpolation provides the smoothest and most stable behavior within the data range. Polynomial interpolation methods are exact at data points but may exhibit oscillatory behavior. Scaling the data improves numerical reliability.

---

## Tools and Technologies
- Python
- NumPy
- Matplotlib
- SciPy

---

## Repository Contents
- `1_task.py` — Curve fitting (Part 1)
- `part2_interpolation.py` — Interpolation methods (Part 2)
- `README.md` — Project report