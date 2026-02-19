"use client";
// components/FileUpload/FileUpload.tsx
// Drag-and-drop VCF file upload with client-side validation

import { useCallback, useRef, useState } from "react";
import { UploadCloud, FileCheck, AlertCircle, X } from "lucide-react";

interface FileUploadProps {
  onFileSelected: (file: File) => void;
  onFileCleared: () => void;
  disabled?: boolean;
}

const MAX_SIZE_BYTES = 5 * 1024 * 1024; // 5 MB

async function validateVcf(file: File): Promise<string | null> {
  // Check extension
  if (!file.name.toLowerCase().endsWith(".vcf")) {
    return "File must have a .vcf extension.";
  }
  // Check size
  if (file.size > MAX_SIZE_BYTES) {
    return `File size ${(file.size / 1e6).toFixed(2)} MB exceeds 5 MB limit.`;
  }
  // Read first 100 bytes and check for ##fileformat=VCF header
  const slice = file.slice(0, 100);
  const text = await slice.text();
  if (!text.startsWith("##fileformat=VCF")) {
    return "Invalid VCF: file must start with ##fileformat=VCF header.";
  }
  return null; // valid
}

export function FileUpload({ onFileSelected, onFileCleared, disabled }: FileUploadProps) {
  const [dragging, setDragging]   = useState(false);
  const [fileName, setFileName]   = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setFileError(null);
      const err = await validateVcf(file);
      if (err) {
        setFileError(err);
        setFileName(null);
        onFileCleared();
        return;
      }
      setFileName(file.name);
      onFileSelected(file);
    },
    [onFileSelected, onFileCleared]
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const clearFile = () => {
    setFileName(null);
    setFileError(null);
    onFileCleared();
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="w-full">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        className={[
          "relative flex flex-col items-center justify-center gap-3",
          "cursor-pointer rounded-xl border-2 border-dashed px-6 py-10",
          "transition-all duration-200",
          disabled
            ? "cursor-not-allowed opacity-50 border-slate-600 bg-slate-800"
            : dragging
            ? "border-cyan-400 bg-cyan-950/30 scale-[1.01]"
            : fileName
            ? "border-emerald-500 bg-emerald-950/30"
            : fileError
            ? "border-red-500 bg-red-950/20"
            : "border-slate-600 bg-slate-800/50 hover:border-cyan-600 hover:bg-slate-800",
        ].join(" ")}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".vcf"
          className="hidden"
          onChange={onInputChange}
          disabled={disabled}
        />

        {fileName ? (
          <>
            <FileCheck className="h-10 w-10 text-emerald-400" />
            <p className="text-sm font-medium text-emerald-300">{fileName}</p>
            <p className="text-xs text-slate-400">VCF file validated ✓</p>
          </>
        ) : fileError ? (
          <>
            <AlertCircle className="h-10 w-10 text-red-400" />
            <p className="text-sm text-red-300 text-center">{fileError}</p>
            <p className="text-xs text-slate-400">Click to try another file</p>
          </>
        ) : (
          <>
            <UploadCloud className="h-10 w-10 text-slate-400" />
            <div className="text-center">
              <p className="text-sm font-medium text-slate-200">
                Drop your VCF file here
              </p>
              <p className="text-xs text-slate-400 mt-1">
                or click to browse — max 5 MB, VCFv4.x
              </p>
            </div>
          </>
        )}
      </div>

      {fileName && (
        <button
          type="button"
          onClick={clearFile}
          className="mt-2 flex items-center gap-1 text-xs text-slate-400 hover:text-red-400 transition-colors"
        >
          <X className="h-3 w-3" /> Remove file
        </button>
      )}
    </div>
  );
}
