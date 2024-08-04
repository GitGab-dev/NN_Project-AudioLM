import torch
import torch.nn.functional as F
import torchaudio
import os
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from pathlib import Path
import csv
from tqdm import tqdm

from SoundStream import audio_to_tokens


class LibriDataset(Dataset):

    def __init__(self, audioDir, newSampleFreq, maxLenght):
        self.audioList = self.getAudioList(audioDir)
        self.resampler = Resample(new_freq=newSampleFreq)
        self.maxLenght = maxLenght

    def getAudioList(self,audioDir):
        flac_files = []
        for root, dirs, files in os.walk(audioDir):
            for file in files:
                if file.endswith(".flac"):
                    file_path = os.path.join(root, file)
                    flac_files.append(file_path)
        return flac_files


    def __len__(self):
        return len(self.audioList)

    def __getitem__(self, idx):
        waveform, sampleRate = torchaudio.load(self.audioList[idx])
        waveformResampled = self.resampler(waveform)
        waveformPadded = F.pad(waveformResampled, pad = (0,self.maxLenght - len(waveformResampled[0])))
        
        return waveformPadded

class TokensDataset(Dataset):

    def __init__(self, rootTokenDir, sampleRate, requiredDuration, includeSemanticTokens = True, includeCoarseTokens = True, includeFineTokens = True):

        self.tokenTypeFlags = (includeSemanticTokens, includeCoarseTokens, includeFineTokens)
        self.rootTokenDir = rootTokenDir
        self.requiredDuration = requiredDuration
        self.sampleRate = sampleRate
        self.tokenList = createTokenList()

    def createTokenList(self):

        ## TO-DO: Use stored tokens to create lists required by flags

        return []

    def __len__(self):
        
        return len(self.tokenList)

    def __getitem__(self, idx):
        
        return tokenList[idx]


def storeTokens(audioDir, outDir, w2vBERT, soundStream, fileCountCheckpoint = 5):

    Path(outDir).mkdir(parents=True, exist_ok=True)

    isNewFile = not os.path.exists(os.path.join(outDir, "out.csv"))

    ## Check for eventual checkpoints
    fileChecked = 0
    reachedCheckpoint = False
    lastFile = None
    
    if os.path.exists(os.path.join(outDir, "checkpoint.txt")):
        with open(os.path.join(outDir, "checkpoint.txt"), mode='r', newline='') as checkpointFile:
            
            fileChecked, lastFile = checkpointFile.readline().strip().split(" ")
            fileChecked = int(fileChecked)
            print("Found a checkpoint!")

    tokenData = []
    fileCount = 0

    totalFiles = 0
    for root, dirs, files in os.walk(audioDir):
        totalFiles += len(files)

    with tqdm(total=totalFiles, desc='Processing files') as pbar:
        for root, dirs, files in os.walk(audioDir):
            for file in files:
    
                reachedCheckpoint = (fileChecked == 0 or file == lastFile or reachedCheckpoint)
                
                if file.endswith(".flac") and reachedCheckpoint and file != lastFile:
              
                    file_path = os.path.join(root, file)
                    waveform, sr = torchaudio.load(file_path)
                    with torch.no_grad():
                        semanticTokens, _ = w2vBERT(waveform)
                        coarseTokens, fineTokens = audio_to_tokens(waveform, sr, soundStream)
                        
                    tokenData.append([file, semanticTokens.tolist(), coarseTokens.tolist(), fineTokens.tolist()])
    
                    fileCount += 1
    
                if fileCount % fileCountCheckpoint == 0 and reachedCheckpoint and file != lastFile:
                    with open(os.path.join(outDir, "out.csv"), mode='a', newline='') as outFile, open(os.path.join(outDir, "checkpoint.txt"), mode='w', newline='') as checkpointFile:
                        writer = csv.writer(outFile, delimiter = ";")
    
                        ## Add header in case of newFile
                        if isNewFile:
                            outFile.write("sep=;\n")
                            writer.writerow(["fileName", "semanticTokens", "coarseTokens", "fineTokens"])
                            isNewFile = not isNewFile
                            
                        writer.writerows(tokenData)
    
                        checkpointFile.write(f"{fileCount + fileChecked} {file}")
                        
                    print(f"SAVED {fileCount} AUDIO ON OUTPUT {os.path.join(outDir, 'out.csv')}. Total of {fileCount + fileChecked} records saved.") 
                    tokenData = []
                    
                pbar.update(1) 

    return fileCount


