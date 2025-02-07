#A polynomial of the form y=p(x), typically for fitting geometric curves in R^2 of this form.

#Header for timestamp and frame id information. Mandatory for an RVIZ2 plugin.
std_msgs/Header header

#The polynomial coefficients of the form y = p[0]*x^(n-1) + p[1]*x^(n-2) + ... + p[n-2]*x + p[n-1]
#I.e. the order numpy's "polyval" and "polyfit" functions use
float64[] polynomial_coefficients

#Row-major covariance matrix on the polynomial coefficients.  Just leave this blank if no covariance data is available
float64[] polynomial_covariance

#The min and max values that can be "safely" extrapolated on this polynomial.
#Typically the range of values that were used to fit the polynomial.
#The RVIZ2 plugin uses these values to select a range of samples to visualize.
float64 xmin 0.0
float64 xmax 0.0