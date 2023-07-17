import os
from glob import glob
from os import environ
from pathlib import Path

import itk
import numpy as np
import tomo_projector as ttc
import argparse
import torch

from data_factory.projectors import CBProjector, add_photon_noise
from tomo_projector_utils.scanner import ConebeamGeometry


def convert_DICOM(folder_path, out_path: str, spacing: float = 2.0):
    PixelType = itk.ctype("signed short")
    Dimension = 3

    ImageType = itk.Image[PixelType, Dimension]
    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(folder_path)
    seriesUID = namesGenerator.GetSeriesUIDs()

    InterpolatorType = itk.LinearInterpolateImageFunction[ImageType, itk.D]
    ResampleFilterType = itk.ResampleImageFilter[ImageType, ImageType]
    InputNamesGeneratorType = itk.GDCMSeriesFileNames
    OutputNamesGeneratorType = itk.NumericSeriesFileNames
    TransformType = itk.IdentityTransform[itk.D, Dimension]

    interpolator = InterpolatorType.New()

    transform = TransformType.New()
    transform.SetIdentity()

    if len(seriesUID) < 1:
        print("No DICOMs in: " + folder_path)
        return False

    for uid in seriesUID:
        seriesIdentifier = uid
        fileNames = namesGenerator.GetFileNames(seriesIdentifier)

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()

        reader.Update()

        inputSpacing = reader.GetOutput().GetSpacing()
        inputRegion = reader.GetOutput().GetLargestPossibleRegion()
        inputSize = inputRegion.GetSize()
        outputSpacing = [spacing, spacing, spacing]

        outputSize = [0, 0, 0]
        outputSize[0] = int(inputSize[0] * inputSpacing[0] / outputSpacing[0] + 0.5)
        outputSize[1] = int(inputSize[1] * inputSpacing[1] / outputSpacing[1] + 0.5)
        outputSize[2] = int(inputSize[2] * inputSpacing[2] / outputSpacing[2] + 0.5)
        transform_type = itk.TranslationTransform[itk.D, 3]
        vector = [0, 0, 0]
        translation = transform_type.New()
        translation.Translate(vector)

        resampler = ResampleFilterType.New()
        resampler.SetInput(reader.GetOutput())
        resampler.SetTransform(translation)
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputOrigin(reader.GetOutput().GetOrigin())
        resampler.SetOutputSpacing(outputSpacing)
        resampler.SetOutputDirection(reader.GetOutput().GetDirection())
        resampler.SetSize(outputSize)
        resampler.Update()

        itk.imwrite(resampler.GetOutput(), out_path)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # setup an argparser to get the data path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to the data folder that contains the 'LIDC-IDRI' folder, defaults to 'data'",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data",
        help="Path to the data folder, defaults to 'data'",
    )
    data_path = Path(parser.parse_args().data_path)
    out_path = Path(parser.parse_args().out_path)
    out_path.mkdir(exist_ok=True, parents=True)

    patient_folders = glob(str(data_path / Path("LIDC-IDRI/*")))

    device = torch.device("cuda")
    torch_rng = torch.Generator(device=device)
    torch_rng = torch_rng.manual_seed(42)
    photon_count = 1e5
    num_projs = 400
    angles = np.linspace(0, 205 / 180 * np.pi, num_projs)
    geom = ConebeamGeometry(
        source_to_center_dst=1000,
        source_to_detector_dst=1500,
        vol_dims=np.array([256, 256, 256]),
        det_dims=np.array([256, 256]),
        vol_spacing=np.array([2, 2, 2]),
        det_spacing=np.array([2, 2]),
        angles=angles,
        det_offset=0,
        sampling_step_size=0.5,
        device=device,
    )
    geom.dump_json("geometry.json")

    i = 0
    # convert from DICOM to itk volume
    for patient_folder in patient_folders:
        case_folders = glob(patient_folder + "/*/*")
        for case_folder in case_folders:
            if i >= 500:
                break
            if len(list(glob(case_folder + "/*"))) > 10:
                convert_DICOM(
                    case_folder,
                    str(out_path / Path(f"volume_{i}_256.mha")),
                    spacing=2.0,
                )
                i = i + 1

    volume_path = sorted(glob("volume_*256.mha"))[0]

    volume = itk.GetArrayFromImage(itk.imread(volume_path))
    volume_cuda = 0.0206 + (0.0206 - 0.0004) * (
        torch.clip(torch.tensor(volume, device="cuda", dtype=torch.float), -1024, 2000)
        / 1000
    )
    volume_cuda = volume_cuda.unsqueeze(0).unsqueeze(0)

    projections = CBProjector.apply(
        volume_cuda,
        *geom.get_projector_params(angles=angles),
    )

    original_volume = volume_cuda[0, 0].cpu().numpy()
    np_projections = projections[0, 0].cpu().numpy()

    np.save("volume.np", original_volume)
    np.save("projections.np", np_projections)

    noisy_projs = (
        add_photon_noise(projections[0, 0], photon_count, torch_rng).cpu().numpy()
    )
    np.save("noisy_projections", noisy_projs)

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    axes = axes.flatten()
    for j, i in enumerate(range(0, num_projs, num_projs // 100)):
        axes[j].imshow(noisy_projs[i])
        axes[j].set_axis_off()
    plt.show()
