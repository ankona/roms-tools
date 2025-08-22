import glob
import logging
import re
import warnings
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import xarray as xr


def _load_data(
    filename,
    dim_names,
    use_dask,
    time_chunking=True,
    decode_times=True,
    force_combine_nested=False,
    read_zarr: bool = False,
):
    """Load dataset from the specified file.

    Parameters
    ----------
    filename : Union[str, Path, List[Union[str, Path]]]
        The path to the data file(s). Can be a single string (with or without wildcards), a single Path object,
        or a list of strings or Path objects containing multiple files.
    dim_names : Dict[str, str], optional
        Dictionary specifying the names of dimensions in the dataset.
        Required only for lat-lon datasets to map dimension names like "latitude" and "longitude".
        For ROMS datasets, this parameter can be omitted, as default ROMS dimensions ("eta_rho", "xi_rho", "s_rho") are assumed.
    use_dask: bool
        Indicates whether to use dask for chunking. If True, data is loaded with dask; if False, data is loaded eagerly. Defaults to False.
    time_chunking : bool, optional
        If True and `use_dask=True`, the data will be chunked along the time dimension with a chunk size of 1.
        If False, the data will not be chunked explicitly along the time dimension, but will follow the default auto chunking scheme. This option is useful for ROMS restart files.
        Defaults to True.
    decode_times: bool, optional
        If True, decode times and timedeltas encoded in the standard NetCDF datetime format into datetime objects. Otherwise, leave them encoded as numbers.
        Defaults to True.
    force_combine_nested: bool, optional
        If True, forces the use of nested combination (`combine_nested`) regardless of whether wildcards are used.
        Defaults to False.
    read_zarr: bool, optional
        If True, use the zarr engine to read the dataset, and don't use mfdataset.
        Defaults to False.

    Returns
    -------
    ds : xr.Dataset
        The loaded xarray Dataset containing the forcing data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If a list of files is provided but dim_names["time"] is not available or use_dask=False.
    """
    if dim_names is None:
        dim_names = {}

    if use_dask:
        if not _has_dask():
            raise RuntimeError(
                "Dask is required but not installed. Install it with:\n"
                "  • `pip install roms-tools[dask]` or\n"
                "  • `conda install dask`\n"
                "Alternatively, install `roms-tools` with conda to include all dependencies."
            )
    if read_zarr:
        if isinstance(filename, list):
            raise ValueError("read_zarr requires a single path, not a list of paths")
        if not use_dask:
            raise ValueError("read_zarr must be used with use_dask")

    # Precompile the regex for matching wildcard characters
    wildcard_regex = re.compile(r"[\*\?\[\]]")

    # Convert Path objects to strings
    if isinstance(filename, (str, Path)):
        filename_str = str(filename)
    elif isinstance(filename, list):
        filename_str = [str(f) for f in filename]
    else:
        raise ValueError("filename must be a string, Path, or a list of strings/Paths.")

    # Handle the case when filename is a string
    contains_wildcard = False
    if isinstance(filename_str, str):
        contains_wildcard = bool(wildcard_regex.search(filename_str))
        if contains_wildcard:
            matching_files = glob.glob(filename_str)
            if not matching_files:
                raise FileNotFoundError(
                    f"No files found matching the pattern '{filename_str}'."
                )
        else:
            matching_files = [filename_str]

    # Handle the case when filename is a list
    elif isinstance(filename_str, list):
        contains_wildcard = any(wildcard_regex.search(f) for f in filename_str)
        if contains_wildcard:
            matching_files = []
            for f in filename_str:
                files = glob.glob(f)
                if not files:
                    raise FileNotFoundError(
                        f"No files found matching the pattern '{f}'."
                    )
                matching_files.extend(files)
        else:
            matching_files = filename_str

    # Sort the matching files
    matching_files = sorted(matching_files)

    # Check if time dimension is available when multiple files are provided
    if isinstance(filename_str, list) and "time" not in dim_names:
        raise ValueError(
            "A list of files is provided, but time dimension is not available. "
            "A time dimension must be available to concatenate the files."
        )

    # Determine the kwargs for combining datasets
    if force_combine_nested:
        kwargs = {"combine": "nested", "concat_dim": dim_names["time"]}
    elif contains_wildcard or len(matching_files) == 1:
        kwargs = {"combine": "by_coords"}
    else:
        kwargs = {"combine": "nested", "concat_dim": dim_names["time"]}

    # Base kwargs used for dataset combination
    combine_kwargs = {
        "coords": "minimal",
        "compat": "override",
        "combine_attrs": "override",
    }

    if use_dask:

        if "latitude" in dim_names and "longitude" in dim_names:
            # for lat-lon datasets
            chunks = {
                dim_names["latitude"]: -1,
                dim_names["longitude"]: -1,
            }
        else:
            # For ROMS datasets
            chunks = {
                "eta_rho": -1,
                "eta_v": -1,
                "xi_rho": -1,
                "xi_u": -1,
                "s_rho": -1,
            }

        if "depth" in dim_names:
            chunks[dim_names["depth"]] = -1
        if "time" in dim_names and time_chunking:
            chunks[dim_names["time"]] = 1
        if "ntides" in dim_names:
            chunks[dim_names["ntides"]] = 1

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=r"^The specified chunks separate.*",
            )

            if read_zarr:
                drops = {
                    "100m_u_component_of_wind",
                    "100m_v_component_of_wind",
                    "10m_u_component_of_neutral_wind",
                    # "10m_u_component_of_wind",
                    "10m_v_component_of_neutral_wind",
                    # "10m_v_component_of_wind",
                    "10m_wind_gust_since_previous_post_processing",
                    # "2m_dewpoint_temperature",
                    # "2m_temperature",
                    "air_density_over_the_oceans",
                    "angle_of_sub_gridscale_orography",
                    "anisotropy_of_sub_gridscale_orography",
                    "benjamin_feir_index",
                    "boundary_layer_dissipation",
                    "boundary_layer_height",
                    "charnock",
                    "clear_sky_direct_solar_radiation_at_surface",
                    "cloud_base_height",
                    "coefficient_of_drag_with_waves",
                    "convective_available_potential_energy",
                    "convective_inhibition",
                    "convective_precipitation",
                    "convective_rain_rate",
                    "convective_snowfall",
                    "convective_snowfall_rate_water_equivalent",
                    "downward_uv_radiation_at_the_surface",
                    "duct_base_height",
                    "eastward_gravity_wave_surface_stress",
                    "eastward_turbulent_surface_stress",
                    "evaporation",
                    "forecast_albedo",
                    "forecast_logarithm_of_surface_roughness_for_heat",
                    "forecast_surface_roughness",
                    "fraction_of_cloud_cover",
                    "free_convective_velocity_over_the_oceans",
                    "friction_velocity",
                    "geopotential",
                    "geopotential_at_surface",
                    "gravity_wave_dissipation",
                    "high_cloud_cover",
                    "high_vegetation_cover",
                    "ice_temperature_layer_1",
                    "ice_temperature_layer_2",
                    "ice_temperature_layer_3",
                    "ice_temperature_layer_4",
                    "instantaneous_10m_wind_gust",
                    "instantaneous_eastward_turbulent_surface_stress",
                    "instantaneous_large_scale_surface_precipitation_fraction",
                    "instantaneous_moisture_flux",
                    "instantaneous_northward_turbulent_surface_stress",
                    "instantaneous_surface_sensible_heat_flux",
                    "k_index",
                    "lake_bottom_temperature",
                    "lake_cover",
                    "lake_depth",
                    "lake_ice_depth",
                    "lake_ice_temperature",
                    "lake_mix_layer_depth",
                    "lake_mix_layer_temperature",
                    "lake_shape_factor",
                    "lake_total_layer_temperature",
                    "land_sea_mask",
                    "large_scale_precipitation",
                    "large_scale_precipitation_fraction",
                    "large_scale_rain_rate",
                    "large_scale_snowfall",
                    "large_scale_snowfall_rate_water_equivalent",
                    "leaf_area_index_high_vegetation",
                    "leaf_area_index_low_vegetation",
                    "low_cloud_cover",
                    "low_vegetation_cover",
                    "maximum_2m_temperature_since_previous_post_processing",
                    "maximum_individual_wave_height",
                    "maximum_total_precipitation_rate_since_previous_post_processing",
                    "mean_boundary_layer_dissipation",
                    "mean_convective_precipitation_rate",
                    "mean_convective_snowfall_rate",
                    "mean_direction_of_total_swell",
                    "mean_direction_of_wind_waves",
                    "mean_eastward_gravity_wave_surface_stress",
                    "mean_eastward_turbulent_surface_stress",
                    "mean_evaporation_rate",
                    "mean_gravity_wave_dissipation",
                    "mean_large_scale_precipitation_fraction",
                    "mean_large_scale_precipitation_rate",
                    "mean_large_scale_snowfall_rate",
                    "mean_northward_gravity_wave_surface_stress",
                    "mean_northward_turbulent_surface_stress",
                    "mean_period_of_total_swell",
                    "mean_period_of_wind_waves",
                    "mean_potential_evaporation_rate",
                    "mean_runoff_rate",
                    "mean_sea_level_pressure",
                    "mean_snow_evaporation_rate",
                    "mean_snowfall_rate",
                    "mean_snowmelt_rate",
                    "mean_square_slope_of_waves",
                    "mean_sub_surface_runoff_rate",
                    "mean_surface_direct_short_wave_radiation_flux",
                    "mean_surface_direct_short_wave_radiation_flux_clear_sky",
                    "mean_surface_downward_long_wave_radiation_flux",
                    "mean_surface_downward_long_wave_radiation_flux_clear_sky",
                    "mean_surface_downward_short_wave_radiation_flux",
                    "mean_surface_downward_short_wave_radiation_flux_clear_sky",
                    "mean_surface_downward_uv_radiation_flux",
                    "mean_surface_latent_heat_flux",
                    "mean_surface_net_long_wave_radiation_flux",
                    "mean_surface_net_long_wave_radiation_flux_clear_sky",
                    "mean_surface_net_short_wave_radiation_flux",
                    "mean_surface_net_short_wave_radiation_flux_clear_sky",
                    "mean_surface_runoff_rate",
                    "mean_surface_sensible_heat_flux",
                    "mean_top_downward_short_wave_radiation_flux",
                    "mean_top_net_long_wave_radiation_flux",
                    "mean_top_net_long_wave_radiation_flux_clear_sky",
                    "mean_top_net_short_wave_radiation_flux",
                    "mean_top_net_short_wave_radiation_flux_clear_sky",
                    "mean_total_precipitation_rate",
                    "mean_vertical_gradient_of_refractivity_inside_trapping_layer",
                    "mean_vertically_integrated_moisture_divergence",
                    "mean_wave_direction",
                    "mean_wave_direction_of_first_swell_partition",
                    "mean_wave_direction_of_second_swell_partition",
                    "mean_wave_direction_of_third_swell_partition",
                    "mean_wave_period",
                    "mean_wave_period_based_on_first_moment",
                    "mean_wave_period_based_on_first_moment_for_swell",
                    "mean_wave_period_based_on_first_moment_for_wind_waves",
                    "mean_wave_period_based_on_second_moment_for_swell",
                    "mean_wave_period_based_on_second_moment_for_wind_waves",
                    "mean_wave_period_of_first_swell_partition",
                    "mean_wave_period_of_second_swell_partition",
                    "mean_wave_period_of_third_swell_partition",
                    "mean_zero_crossing_wave_period",
                    "medium_cloud_cover",
                    "minimum_2m_temperature_since_previous_post_processing",
                    "minimum_total_precipitation_rate_since_previous_post_processing",
                    "minimum_vertical_gradient_of_refractivity_inside_trapping_layer",
                    "model_bathymetry",
                    "near_ir_albedo_for_diffuse_radiation",
                    "near_ir_albedo_for_direct_radiation",
                    "normalized_energy_flux_into_ocean",
                    "normalized_energy_flux_into_waves",
                    "normalized_stress_into_ocean",
                    "northward_gravity_wave_surface_stress",
                    "northward_turbulent_surface_stress",
                    "ocean_surface_stress_equivalent_10m_neutral_wind_direction",
                    "ocean_surface_stress_equivalent_10m_neutral_wind_speed",
                    "ozone_mass_mixing_ratio",
                    "peak_wave_period",
                    "period_corresponding_to_maximum_individual_wave_height",
                    "potential_evaporation",
                    "potential_vorticity",
                    "precipitation_type",
                    "runoff",
                    "sea_ice_cover",
                    # "sea_surface_temperature",
                    "significant_height_of_combined_wind_waves_and_swell",
                    "significant_height_of_total_swell",
                    "significant_height_of_wind_waves",
                    "significant_wave_height_of_first_swell_partition",
                    "significant_wave_height_of_second_swell_partition",
                    "significant_wave_height_of_third_swell_partition",
                    "skin_reservoir_content",
                    "skin_temperature",
                    "slope_of_sub_gridscale_orography",
                    "snow_albedo",
                    "snow_density",
                    "snow_depth",
                    "snow_evaporation",
                    "snowfall",
                    "snowmelt",
                    "soil_temperature_level_1",
                    "soil_temperature_level_2",
                    "soil_temperature_level_3",
                    "soil_temperature_level_4",
                    "soil_type",
                    "specific_cloud_ice_water_content",
                    "specific_cloud_liquid_water_content",
                    "specific_humidity",
                    "standard_deviation_of_filtered_subgrid_orography",
                    "standard_deviation_of_orography",
                    "sub_surface_runoff",
                    "surface_latent_heat_flux",
                    # "surface_net_solar_radiation",
                    "surface_net_solar_radiation_clear_sky",
                    "surface_net_thermal_radiation",
                    "surface_net_thermal_radiation_clear_sky",
                    "surface_pressure",
                    "surface_runoff",
                    "surface_sensible_heat_flux",
                    "surface_solar_radiation_downward_clear_sky",
                    "surface_solar_radiation_downwards",
                    "surface_thermal_radiation_downward_clear_sky",
                    # "surface_thermal_radiation_downwards",
                    "temperature",
                    "temperature_of_snow_layer",
                    "toa_incident_solar_radiation",
                    "top_net_solar_radiation",
                    "top_net_solar_radiation_clear_sky",
                    "top_net_thermal_radiation",
                    "top_net_thermal_radiation_clear_sky",
                    "total_cloud_cover",
                    "total_column_cloud_ice_water",
                    "total_column_cloud_liquid_water",
                    "total_column_ozone",
                    "total_column_rain_water",
                    "total_column_snow_water",
                    "total_column_supercooled_liquid_water",
                    "total_column_water",
                    "total_column_water_vapour",
                    # "total_precipitation",
                    "total_sky_direct_solar_radiation_at_surface",
                    "total_totals_index",
                    "trapping_layer_base_height",
                    "trapping_layer_top_height",
                    "type_of_high_vegetation",
                    "type_of_low_vegetation",
                    "u_component_of_wind",
                    "u_component_stokes_drift",
                    "uv_visible_albedo_for_diffuse_radiation",
                    "uv_visible_albedo_for_direct_radiation",
                    "v_component_of_wind",
                    "v_component_stokes_drift",
                    "vertical_integral_of_divergence_of_cloud_frozen_water_flux",
                    "vertical_integral_of_divergence_of_cloud_liquid_water_flux",
                    "vertical_integral_of_divergence_of_geopotential_flux",
                    "vertical_integral_of_divergence_of_kinetic_energy_flux",
                    "vertical_integral_of_divergence_of_mass_flux",
                    "vertical_integral_of_divergence_of_moisture_flux",
                    "vertical_integral_of_divergence_of_ozone_flux",
                    "vertical_integral_of_divergence_of_thermal_energy_flux",
                    "vertical_integral_of_divergence_of_total_energy_flux",
                    "vertical_integral_of_eastward_cloud_frozen_water_flux",
                    "vertical_integral_of_eastward_cloud_liquid_water_flux",
                    "vertical_integral_of_eastward_geopotential_flux",
                    "vertical_integral_of_eastward_heat_flux",
                    "vertical_integral_of_eastward_kinetic_energy_flux",
                    "vertical_integral_of_eastward_mass_flux",
                    "vertical_integral_of_eastward_ozone_flux",
                    "vertical_integral_of_eastward_total_energy_flux",
                    "vertical_integral_of_eastward_water_vapour_flux",
                    "vertical_integral_of_energy_conversion",
                    "vertical_integral_of_kinetic_energy",
                    "vertical_integral_of_mass_of_atmosphere",
                    "vertical_integral_of_mass_tendency",
                    "vertical_integral_of_northward_cloud_frozen_water_flux",
                    "vertical_integral_of_northward_cloud_liquid_water_flux",
                    "vertical_integral_of_northward_geopotential_flux",
                    "vertical_integral_of_northward_heat_flux",
                    "vertical_integral_of_northward_kinetic_energy_flux",
                    "vertical_integral_of_northward_mass_flux",
                    "vertical_integral_of_northward_ozone_flux",
                    "vertical_integral_of_northward_total_energy_flux",
                    "vertical_integral_of_northward_water_vapour_flux",
                    "vertical_integral_of_potential_and_internal_energy",
                    "vertical_integral_of_potential_internal_and_latent_energy",
                    "vertical_integral_of_temperature",
                    "vertical_integral_of_thermal_energy",
                    "vertical_integral_of_total_energy",
                    "vertical_velocity",
                    "vertically_integrated_moisture_divergence",
                    "volumetric_soil_water_layer_1",
                    "volumetric_soil_water_layer_2",
                    "volumetric_soil_water_layer_3",
                    "volumetric_soil_water_layer_4",
                    "wave_spectral_directional_width",
                    "wave_spectral_directional_width_for_swell",
                    "wave_spectral_directional_width_for_wind_waves",
                    "wave_spectral_kurtosis",
                    "wave_spectral_peakedness",
                    "wave_spectral_skewness",
                    "zero_degree_level",
                    # "latitude",
                    # "level",
                    # "longitude",
                    # "time",
                }
                ds = xr.open_zarr(
                    matching_files[0],
                    decode_times=decode_times,
                    chunks=chunks,
                    consolidated=None,
                    storage_options=dict(token="anon"),
                    # drop_variables=drops,
                )
            else:
                ds = xr.open_mfdataset(
                    matching_files,
                    decode_times=decode_times,
                    decode_timedelta=decode_times,
                    chunks=chunks,
                    **combine_kwargs,
                    **kwargs,
                )

    else:
        ds_list = []
        for file in matching_files:
            ds = xr.open_dataset(
                file,
                decode_times=decode_times,
                decode_timedelta=decode_times,
                chunks=None,
            )
            ds_list.append(ds)

        if kwargs["combine"] == "by_coords":
            ds = xr.combine_by_coords(ds_list, **combine_kwargs)
        elif kwargs["combine"] == "nested":
            ds = xr.combine_nested(
                ds_list, concat_dim=kwargs["concat_dim"], **combine_kwargs
            )

    if "time" in dim_names and dim_names["time"] not in ds.dims:
        ds = ds.expand_dims(dim_names["time"])

    if "time" in dim_names and not read_zarr:
        ds = ds.drop_duplicates(dim=dim_names["time"])

    return ds


def interpolate_from_rho_to_u(field, method="additive"):
    """Interpolates the given field from rho points to u points.

    This function performs an interpolation from the rho grid (cell centers) to the u grid
    (cell edges in the xi direction). Depending on the chosen method, it either averages
    (additive) or multiplies (multiplicative) the field values between adjacent rho points
    along the xi dimension. It also handles the removal of unnecessary coordinate variables
    and updates the dimensions accordingly.

    Parameters
    ----------
    field : xr.DataArray
        The input data array on the rho grid to be interpolated. It is assumed to have a dimension
        named "xi_rho".

    method : str, optional, default='additive'
        The method to use for interpolation. Options are:
        - 'additive': Average the field values between adjacent rho points.
        - 'multiplicative': Multiply the field values between adjacent rho points. Appropriate for
          binary masks.

    Returns
    -------
    field_interpolated : xr.DataArray
        The interpolated data array on the u grid with the dimension "xi_u".
    """

    if method == "additive":
        field_interpolated = 0.5 * (field + field.shift(xi_rho=1)).isel(
            xi_rho=slice(1, None)
        )
    elif method == "multiplicative":
        field_interpolated = (field * field.shift(xi_rho=1)).isel(xi_rho=slice(1, None))
    else:
        raise NotImplementedError(f"Unsupported method '{method}' specified.")

    vars_to_drop = ["lat_rho", "lon_rho", "eta_rho", "xi_rho"]
    for var in vars_to_drop:
        if var in field_interpolated.coords:
            field_interpolated = field_interpolated.drop_vars(var)

    field_interpolated = field_interpolated.swap_dims({"xi_rho": "xi_u"})

    return field_interpolated


def interpolate_from_rho_to_v(field, method="additive"):
    """Interpolates the given field from rho points to v points.

    This function performs an interpolation from the rho grid (cell centers) to the v grid
    (cell edges in the eta direction). Depending on the chosen method, it either averages
    (additive) or multiplies (multiplicative) the field values between adjacent rho points
    along the eta dimension. It also handles the removal of unnecessary coordinate variables
    and updates the dimensions accordingly.

    Parameters
    ----------
    field : xr.DataArray
        The input data array on the rho grid to be interpolated. It is assumed to have a dimension
        named "eta_rho".

    method : str, optional, default='additive'
        The method to use for interpolation. Options are:
        - 'additive': Average the field values between adjacent rho points.
        - 'multiplicative': Multiply the field values between adjacent rho points. Appropriate for
          binary masks.

    Returns
    -------
    field_interpolated : xr.DataArray
        The interpolated data array on the v grid with the dimension "eta_v".
    """

    if method == "additive":
        field_interpolated = 0.5 * (field + field.shift(eta_rho=1)).isel(
            eta_rho=slice(1, None)
        )
    elif method == "multiplicative":
        field_interpolated = (field * field.shift(eta_rho=1)).isel(
            eta_rho=slice(1, None)
        )
    else:
        raise NotImplementedError(f"Unsupported method '{method}' specified.")

    vars_to_drop = ["lat_rho", "lon_rho", "eta_rho", "xi_rho"]
    for var in vars_to_drop:
        if var in field_interpolated.coords:
            field_interpolated = field_interpolated.drop_vars(var)

    field_interpolated = field_interpolated.swap_dims({"eta_rho": "eta_v"})

    return field_interpolated


def transpose_dimensions(da: xr.DataArray) -> xr.DataArray:
    """Transpose the dimensions of an xarray.DataArray to ensure that 'time', any
    dimension starting with 's_', 'eta_', and 'xi_' are ordered first, followed by the
    remaining dimensions in their original order.

    Parameters
    ----------
    da : xarray.DataArray
        The input DataArray whose dimensions are to be reordered.

    Returns
    -------
    xarray.DataArray
        The DataArray with dimensions reordered so that 'time', 's_*', 'eta_*',
        and 'xi_*' are first, in that order, if they exist.
    """

    # List of preferred dimension patterns
    preferred_order = ["time", "s_", "eta_", "xi_"]

    # Get the existing dimensions in the DataArray
    dims = list(da.dims)

    # Collect dimensions that match any of the preferred patterns
    matched_dims = []
    for pattern in preferred_order:
        # Find dimensions that start with the pattern
        matched_dims += [dim for dim in dims if dim.startswith(pattern)]

    # Create a new order: first the matched dimensions, then the rest
    remaining_dims = [dim for dim in dims if dim not in matched_dims]
    new_order = matched_dims + remaining_dims

    # Transpose the DataArray to the new order
    transposed_da = da.transpose(*new_order)

    return transposed_da


def save_datasets(dataset_list, output_filenames, use_dask=False, verbose=True):
    """Save the list of datasets to netCDF4 files.

    Parameters
    ----------
    dataset_list : list
        List of datasets to be saved.
    output_filenames : list
        List of filenames for the output files.
    use_dask : bool, optional
        Whether to use Dask diagnostics (e.g., progress bars) when saving the datasets, by default False.
    verbose : bool, optional
        Whether to log information about the files being written. If True, logs the output filenames.
        Defaults to True.

    Returns
    -------
    List[Path]
        A list of Path objects for the filenames that were saved.
    """

    saved_filenames = []

    output_filenames = [f"{filename}.nc" for filename in output_filenames]
    if verbose:
        logging.info(
            "Writing the following NetCDF files:\n%s", "\n".join(output_filenames)
        )

    if use_dask:
        from dask.diagnostics import ProgressBar

        with ProgressBar():
            xr.save_mfdataset(dataset_list, output_filenames)
    else:
        xr.save_mfdataset(dataset_list, output_filenames)

    saved_filenames.extend(Path(f) for f in output_filenames)

    return saved_filenames


def get_dask_chunks(location, chunk_size):
    """Returns the appropriate Dask chunking dictionary based on grid location.

    Parameters
    ----------
    location : str
        The grid location, one of "rho", "u", or "v".
    chunk_size : int
        The chunk size to apply.

    Returns
    -------
    dict
        Dictionary specifying the chunking strategy.
    """
    chunk_mapping = {
        "rho": {"eta_rho": chunk_size, "xi_rho": chunk_size},
        "u": {"eta_rho": chunk_size, "xi_u": chunk_size},
        "v": {"eta_v": chunk_size, "xi_rho": chunk_size},
    }
    return chunk_mapping.get(location, {})


def _generate_coordinate_range(min, max, resolution):
    """Generate an array of target coordinates (e.g., latitude or longitude) within a
    specified range, with a resolution that is rounded to the nearest value of the form
    `1/n` (or integer).

    This method generates an array of target coordinates between the provided `min` and `max`
    values, ensuring that both `min` and `max` are included in the resulting range. The resolution
    is rounded to the nearest fraction of the form `1/n` or an integer, based on the input.

    Parameters
    ----------
    min : float
        The minimum value (in degrees) of the coordinate range (inclusive).

    max : float
        The maximum value (in degrees) of the coordinate range (inclusive).

    resolution : float
        The spacing (in degrees) between each coordinate in the array. The resolution will
        be rounded to the nearest value of the form `1/n` or an integer, depending on the size
        of the resolution value.

    Returns
    -------
    numpy.ndarray
        An array of target coordinates generated from the specified range, with the resolution
        rounded to a suitable fraction (e.g., `1/n`) or integer, depending on the input resolution.
    """

    # Find the closest fraction of the form 1/n or integer to match the resolution
    resolution_rounded = None
    min_diff = float("inf")  # Initialize the minimum difference as infinity

    # Search for the best fraction or integer approximation to the resolution
    for n in range(1, 1000):  # Try fractions 1/n, where n ranges from 1 to 999
        if resolution <= 1:
            fraction = (
                1 / n
            )  # For small resolutions (<= 1), consider fractions of the form 1/n
        else:
            fraction = n  # For larger resolutions (>1), consider integers (n)

        diff = abs(
            fraction - resolution
        )  # Calculate the difference between the fraction and the resolution

        if diff < min_diff:  # If the current fraction is a better approximation
            min_diff = diff
            resolution_rounded = fraction  # Update the best fraction (or integer) found

    # Adjust the start and end of the range to include integer values
    start_int = np.floor(min)  # Round the minimum value down to the nearest integer
    end_int = np.ceil(max)  # Round the maximum value up to the nearest integer

    # Generate the array of target coordinates, including both the min and max values
    target = np.arange(start_int, end_int + resolution_rounded, resolution_rounded)

    # Truncate any values that exceed max (including small floating point errors)
    target = target[target <= end_int + 1e-10]

    return target.astype(np.float32)


def _remove_edge_nans(field, xdim, layer_depth=None):
    """Trim NaN-only edges along a specified dimension.

    Useful when a ROMS grid has been regridded to a fixed lat/lon section,
    leaving NaN-filled edges (e.g., over land or outside the domain).
    Removes leading/trailing slices along `xdim` where all values are NaN,
    based on `field` or optionally `layer_depth`.

    Parameters
    ----------
    field : xr.DataArray
        Data to trim.
    xdim : str
        Dimension along which to remove NaN edges.
    layer_depth : xr.DataArray, optional
        Optional field to determine where NaNs occur.

    Returns
    -------
    field : xr.DataArray
        Trimmed data.
    layer_depth : xr.DataArray or None
        Trimmed `layer_depth` if provided.
    """
    if xdim in field.dims:
        if layer_depth is not None:
            nan_mask = layer_depth.isnull().sum(
                dim=[dim for dim in layer_depth.dims if dim != xdim]
            )
        else:
            nan_mask = field.isnull().sum(
                dim=[dim for dim in field.dims if dim != xdim]
            )

        # Find the valid indices where the sum of the nans is 0
        valid_indices = np.where(nan_mask.values == 0)[0]

        if len(valid_indices) > 0:
            first_valid = valid_indices[0]
            last_valid = valid_indices[-1]

            field = field.isel({xdim: slice(first_valid, last_valid + 1)})
            if layer_depth is not None:
                layer_depth = layer_depth.isel(
                    {xdim: slice(first_valid, last_valid + 1)}
                )

    return field, layer_depth


def _has_dask() -> bool:
    return find_spec("dask") is not None


def _has_gcsfs() -> bool:
    return find_spec("gcsfs") is not None


def normalize_longitude(lon: float, straddle: bool) -> float:
    """Normalize longitude to the appropriate range depending on whether the grid
    straddles the dateline.

    Parameters
    ----------
    lon : float
        Longitude in degrees (can be any value, including multiples of 360 or negative values).
    straddle : bool
        Whether the grid straddles the dateline. If True, output will be in (-180, 180];
        if False, output will be in [0, 360).
    Returns
    -------
    float
        Normalized longitude.
    """
    lon = lon % 360
    if straddle:
        return lon - 360 if lon > 180 else lon
    else:
        return lon + 360 if lon < 0 else lon
