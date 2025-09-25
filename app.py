import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

@st.cache_data
def load_data():
    years = np.arange(1960, 2021)

    data = {
        "Year": years,
        "Argentina_Population": np.linspace(20, 45, len(years)),  
        "Brazil_Population": np.linspace(72, 213, len(years)),   
        "Mexico_Population": np.linspace(38, 129, len(years)),   
        "Argentina_LifeExp": np.linspace(63, 77, len(years)),
        "Brazil_LifeExp": np.linspace(54, 75, len(years)),
        "Mexico_LifeExp": np.linspace(57, 76, len(years)),
        "Argentina_GDPpc": np.linspace(3000, 10000, len(years)),  
        "Brazil_GDPpc": np.linspace(2000, 9000, len(years)),
        "Mexico_GDPpc": np.linspace(2500, 9500, len(years)),
        "Argentina_BirthRate": np.linspace(30, 17, len(years)),
        "Brazil_BirthRate": np.linspace(42, 14, len(years)),
        "Mexico_BirthRate": np.linspace(45, 18, len(years)),
        "Argentina_Homicide": np.linspace(5, 6, len(years)),
        "Brazil_Homicide": np.linspace(10, 27, len(years)),
        "Mexico_Homicide": np.linspace(5, 29, len(years)),
    }
    df = pd.DataFrame(data)
    return df

df = load_data()

st.title("Latin America Historical Data Regression Explorer")

entity = st.selectbox(
    "Select indicator:",
    [
        "Population",
        "Life Expectancy",
        "Average Income (GDP per capita)",
        "Birth Rate",
        "Murder Rate",
    ],
)

countries = st.multiselect(
    "Select countries:",
    ["Argentina", "Brazil", "Mexico"],
    default=["Argentina"],
)

degree = st.slider("Polynomial degree:", 3, 6, 3)
step = st.slider("Graph increments (years):", 1, 10, 1)
extra_years = st.slider("Extrapolate into the future (years):", 0, 50, 20)

st.subheader("Raw Data (editable)")
editable_df = st.data_editor(df, num_rows="dynamic")

def fit_and_plot(x, y, label):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression().fit(X_poly, y)
    coef = model.coef_
    intercept = model.intercept_

    terms = [f"{coef[i]:.4e}x^{i}" for i in range(1, len(coef))]
    eqn = f"y = {intercept:.4e} + " + " + ".join(terms)

    x_future = np.arange(x.min(), x.max() + extra_years + 1, step)
    X_future = poly.transform(x_future.reshape(-1, 1))
    y_future = model.predict(X_future)

    plt.scatter(x, y, label=f"{label} data")
    plt.plot(x_future, y_future, label=f"{label} fit")
    if extra_years > 0:
        plt.axvline(x.max(), color="gray", linestyle="--")
        plt.plot(x_future[x_future > x.max()], 
                 y_future[x_future > x.max()],
                 linestyle="--", color="red", label=f"{label} extrapolated")

    return model, poly, eqn, x_future, y_future

entity_map = {
    "Population": "Population",
    "Life Expectancy": "LifeExp",
    "Average Income (GDP per capita)": "GDPpc",
    "Birth Rate": "BirthRate",
    "Murder Rate": "Homicide",
}
indicator = entity_map[entity]

st.subheader("Regression Analysis")
fig = plt.figure()
results = {}
for country in countries:
    col = f"{country}_{indicator}"
    x = editable_df["Year"].values
    y = editable_df[col].values
    model, poly, eqn, xf, yf = fit_and_plot(x, y, country)
    results[country] = (model, poly, eqn, xf, yf)

plt.xlabel("Year")
plt.ylabel(entity)
plt.legend()
st.pyplot(fig)

for country, (_, _, eqn, _, _) in results.items():
    st.markdown(f"**{country} regression equation:** {eqn}")

st.subheader("Function Analysis")
for country, (model, poly, eqn, xf, yf) in results.items():
    dy = np.gradient(yf, step)
    max_rate_idx = np.argmax(dy)
    min_rate_idx = np.argmin(dy)

    st.markdown(f"### {country}")
    st.write(f"- Local fastest increase around year {int(xf[max_rate_idx])}.")
    st.write(f"- Local fastest decrease around year {int(xf[min_rate_idx])}.")
    st.write(f"- Domain: {xf.min()}–{xf.max()} years.")
    st.write(f"- Range: {yf.min():.2f}–{yf.max():.2f} {entity} units.")
    st.write(f"- Extrapolated {entity} in {int(xf.max())}: {yf[-1]:.2f} units.")

st.subheader("Interpolation / Extrapolation Tool")
input_year = st.number_input("Enter year:", 1960, 2100, 2030)
for country, (model, poly, eqn, _, _) in results.items():
    x_in = np.array([[input_year]])
    x_poly = poly.transform(x_in)
    y_pred = model.predict(x_poly)[0]
    st.write(f"{country} {entity} in {input_year}: {y_pred:.2f}")

st.subheader("Average Rate of Change")
year1 = st.number_input("Start year:", 1960, 2100, 1960)
year2 = st.number_input("End year:", 1960, 2100, 2000)
for country, (model, poly, eqn, _, _) in results.items():
    x1 = poly.transform(np.array([[year1]]))
    x2 = poly.transform(np.array([[year2]]))
    y1 = model.predict(x1)[0]
    y2 = model.predict(x2)[0]
    avg_rate = (y2 - y1) / (year2 - year1)
    st.write(f"{country}: Avg rate of change between {year1}-{year2} = {avg_rate:.2f} units/year")

st.subheader("Printer-Friendly Report")
if st.button("Generate Printer-Friendly Report"):
    st.write("## Regression Report")
    for country, (_, _, eqn, xf, yf) in results.items():
        st.write(f"### {country}")
        st.write(f"Equation: {eqn}")
        st.write(f"Domain: {xf.min()}–{xf.max()}")
        st.write(f"Range: {yf.min():.2f}–{yf.max():.2f}")
