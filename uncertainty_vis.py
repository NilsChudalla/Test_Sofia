import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

logo_URL = 'https://raw.githubusercontent.com/NilsChudalla/Test_Sofia/main/CGRE_logo.svg'
logo_URL2 = 'https://raw.githubusercontent.com/NilsChudalla/Test_Sofia/main/PLUS_logo.png'
ent_URL = 'https://raw.githubusercontent.com/NilsChudalla/Test_Sofia/main/entropy_block.csv'
prob_URL = 'https://raw.githubusercontent.com/NilsChudalla/Test_Sofia/main/prob_block2.csv'
uncert_URL = 'https://raw.githubusercontent.com/NilsChudalla/Test_Sofia/main/Uncertainty_table.csv'

lith_dict= {"Upper Freshwater Molasse": 0,
            "Upper Marine Molasse": 1,
            "Mid-Lower Freshwater Molasse": 2,
            "Lower Freshwater Molasse": 3,
            "Lower Marine Molasse": 4,
            "Mesozoic": 5,
            "Basement": 6}

head1, head2, head3= st.columns(3)
head1.image(logo_URL, use_column_width=True)
head3.image(logo_URL2, use_column_width=True)

# set title
#st.set_page_config(layout="wide")

st.title('Uncertainty in triangle zones')
st.markdown('''This is a simple visualization of uncertainties in the model presented by Sofia Brisson. 
It contains options to view either probabilities or information entropy. These were calculated using the open source 
modelling library **Gempy**.''')
data_type = st.sidebar.radio("Uncertainty style",
         ('Probability', 'Entropy'))

st.subheader('Uncertainty input')
uncert_df = pd.read_csv(uncert_URL, delimiter=';', index_col=[0])
st.table(uncert_df)

# set data source




# load data into cache
resolution = np.array([50, 50, 50])
@st.cache
def load_data():
    prob_block = pd.read_csv(prob_URL, index_col=[0])
    ent_block = pd.read_csv(ent_URL, usecols=['vals']).values.reshape(resolution)

    #block = np.load(DATA_URL, allow_pickle=True).reshape(resolution)
    return prob_block, ent_block
# info text for loading data

prob_block, ent_block = load_data()


# Setting plotting parameters
xmin, xmax, ymin, ymax, zmin, zmax = [6, 25, 3, 35, -5.5, 0.5]

d_grid = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
cell_width = d_grid / resolution
grid_min = np.array([xmin, ymin, zmin]) + cell_width/0.5
grid_max = np.array([xmax, ymax, zmax]) - cell_width/0.5

x_array = np.linspace(grid_min[0], grid_max[0], resolution[0].astype(int))
y_array = np.linspace(grid_min[1], grid_max[1], resolution[1].astype(int))
z_array = np.linspace(grid_min[2], grid_max[2], resolution[2].astype(int))

XX_xy, YY_xy = np.meshgrid(x_array, y_array)
XX_xz, ZZ_xz = np.meshgrid(x_array, z_array)
YY_yz, ZZ_yz = np.meshgrid(y_array, z_array)


z_label = np.linspace(-5.5, 0.5, 7)

depth = float(np.mean(z_array))
WE_profile = float(np.mean(y_array))
NS_profile = float(np.mean(x_array))


depth = st.sidebar.slider(label='Depth slider', min_value=float(np.min(z_array)), max_value=float(np.max(z_array)),
                      value=float(np.mean(z_array)))
depth_index = np.argmin(np.abs(z_array - depth))
WE_profile = st.sidebar.slider(label='Profile slider (W-E)', min_value=float(np.min(y_array)), max_value=float(np.max(y_array)),
                       value=float(np.mean(y_array)))
WE_index = np.argmin(np.abs(y_array - WE_profile))
NS_profile = st.sidebar.slider(label='Profile slider (N-S)', min_value=float(np.min(x_array)), max_value=float(np.max(x_array)),
                       value=float(np.mean(x_array)))
NS_index = np.argmin(np.abs(x_array - NS_profile))
cross_sections = st.sidebar.checkbox('Toggle relative profile positions')
super_elevation = st.sidebar.slider(label='Vertical exaggeration', min_value=1, max_value=4, value=2)


if data_type == 'Probability':
    choose_lith = st.sidebar.selectbox("Select lithology:", options=["Upper Freshwater Molasse", "Upper Marine Molasse",
                                                                       "Mid-Lower Freshwater Molasse", "Lower Freshwater Molasse",
                                                                       "Lower Marine Molasse", "Mesozoic", "Basement"])
    prob_solution = lith_dict[choose_lith]
    block = prob_block[prob_block.columns[prob_solution]].values.reshape(resolution)
    title = ('Probability visualization - ' + choose_lith)
    curr_cmap = 'viridis'
    vmin = 0.0
    vmax = 1.0

elif data_type == 'Entropy':
    block = ent_block
    title = 'Entropy visualization'
    curr_cmap = 'magma'
    vmin = 0.0
    vmax = 1.0

fig0=plt.figure(figsize=(6,0.7))
fig0.patch.set_facecolor("#8EBAE5")

vals = np.linspace(0,1)

vals = np.vstack((vals, vals))
plt.imshow(vals, cmap=curr_cmap, extent=[0, 1, 0, 0.15])
if data_type == "Entropy":
    plt.xlabel('Shannon cell Entropy')

else:
    plt.xlabel('Probability')
ax = plt.gca()
ax.axes.get_yaxis().set_ticks([])
st.sidebar.pyplot(fig0)


st.subheader(title)

with st.expander('Depthmap'):

    hslice_coords = block[:, :, depth_index]

    fig1 = plt.figure(figsize=(5,5))
    AX = gridspec.GridSpec(1,1)
    AX.update(wspace = 0.1, hspace = 0.5)
    ax1 = plt.subplot(AX[0,0])

    ax1.set_title('Depthmap')

    ax1.imshow(hslice_coords, extent=[grid_min[1], grid_max[1], grid_min[0], grid_max[0]], origin='lower', cmap=curr_cmap, vmin=vmin, vmax=vmax)
    #ax1.axis('equal')

    if cross_sections:
        ax1.vlines(WE_profile, ymin=grid_min[0], ymax=grid_max[0], color='#a32632', lw=1)
        ax1.hlines(NS_profile, xmin=grid_min[1], xmax=grid_max[1], color='#a32632', lw=1)
    st.pyplot(fig1)

with st.expander('N-S profile'):

    WE_slice = block[:, WE_index, :]

    fig2 = plt.figure()
    AX = gridspec.GridSpec(1,1)
    AX.update(wspace = 0.1, hspace = 0.5)

    ax2 = plt.subplot(AX[0,0])
    ax2.set_title('N-S profile')
    #ax2.contourf(XX_xz, ZZ_xz, WE_slice.T, cmap='magma', vmin=vmin, vmax=vmax)

    ax2.imshow(WE_slice.T, extent=[grid_min[0], grid_max[0], grid_min[2] * super_elevation, grid_max[2] * super_elevation], vmin=vmin, vmax=vmax, cmap=curr_cmap, origin='lower')

    ax2.yaxis.set_ticks(np.linspace(grid_min[2] * super_elevation, grid_max[2] * super_elevation, 7))
    ax2.yaxis.set_ticklabels(z_label)

#ax2.axis('equal')

#ax2.set_ylim(np.min(z_array), np.max(z_array))

    if cross_sections:
        ax2.vlines(NS_profile, ymin=grid_min[2] * super_elevation, ymax=grid_max[2] * super_elevation, color='#a32632', lw=1)
        ax2.hlines(depth * super_elevation, xmin=grid_min[0], xmax=grid_max[0], color='#a32632', lw=1)
    st.pyplot(fig2)

with st.expander('W-E profile'):

    NS_slice = block[NS_index, :, :]
    fig3 = plt.figure()
    AX = gridspec.GridSpec(1, 1)
    AX.update(wspace=0.1, hspace=0.5)

    ax3 = plt.subplot(AX[0,0])
    ax3.set_title('W-E profile')
    ax3.imshow(NS_slice.T, cmap=curr_cmap, extent=[grid_min[1], grid_max[1], grid_min[2] * super_elevation, grid_max[2] * super_elevation], vmin=vmin, vmax=vmax, origin='lower')
    labels = [item.get_text() for item in ax2.get_yticklabels()]
    ax3.yaxis.set_ticks(np.linspace(grid_min[2] * super_elevation, grid_max[2] * super_elevation, 7))
    ax3.yaxis.set_ticklabels(z_label)
    ax3.invert_xaxis()

    if cross_sections:
        ax3.vlines(WE_profile, ymin=grid_min[2] * super_elevation, ymax= grid_max[2] * super_elevation, color='#a32632', lw=1)
        ax3.hlines(depth * super_elevation, xmin=grid_min[1], xmax=grid_max[1], color='#a32632', lw=1)
    st.pyplot(fig3)