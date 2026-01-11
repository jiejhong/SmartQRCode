#include <jni.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <android/log.h>

#include "BarcodeFormat.h"
#include "ReadBarcode.h"
#include "ReaderOptions.h"
#include "ByteArray.h"
#include "Error.h"

namespace {

struct RectI {
	int left = 0;
	int top = 0;
	int right = 0;
	int bottom = 0;
};

static RectI clampRect(RectI r, int width, int height)
{
	r.left = std::clamp(r.left, 0, width);
	r.right = std::clamp(r.right, 0, width);
	r.top = std::clamp(r.top, 0, height);
	r.bottom = std::clamp(r.bottom, 0, height);
	if (r.right < r.left)
		std::swap(r.right, r.left);
	if (r.bottom < r.top)
		std::swap(r.bottom, r.top);
	return r;
}

static std::vector<uint8_t> cropGray(const uint8_t* src, int width, int height, RectI roi, int& outW, int& outH)
{
	roi = clampRect(roi, width, height);
	outW = std::max(0, roi.right - roi.left);
	outH = std::max(0, roi.bottom - roi.top);
	std::vector<uint8_t> dst(static_cast<size_t>(outW) * static_cast<size_t>(outH));
	for (int y = 0; y < outH; y++) {
		const uint8_t* row = src + (roi.top + y) * width + roi.left;
		std::copy(row, row + outW, dst.data() + static_cast<size_t>(y) * outW);
	}
	return dst;
}

static std::vector<uint8_t> rotateGray(const uint8_t* src, int width, int height, int rotationDeg, int& outW, int& outH)
{
	const int rot = ((rotationDeg % 360) + 360) % 360;
	if (rot == 0) {
		outW = width;
		outH = height;
		return std::vector<uint8_t>(src, src + static_cast<size_t>(width) * static_cast<size_t>(height));
	}

	if (rot == 180) {
		outW = width;
		outH = height;
		std::vector<uint8_t> dst(static_cast<size_t>(width) * static_cast<size_t>(height));
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				dst[static_cast<size_t>((height - 1 - y) * width + (width - 1 - x))] = src[static_cast<size_t>(y * width + x)];
			}
		}
		return dst;
	}

	if (rot == 90 || rot == 270) {
		outW = height;
		outH = width;
		std::vector<uint8_t> dst(static_cast<size_t>(outW) * static_cast<size_t>(outH));
		if (rot == 90) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					const int nx = height - 1 - y;
					const int ny = x;
					dst[static_cast<size_t>(ny * outW + nx)] = src[static_cast<size_t>(y * width + x)];
				}
			}
		} else {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					const int nx = y;
					const int ny = width - 1 - x;
					dst[static_cast<size_t>(ny * outW + nx)] = src[static_cast<size_t>(y * width + x)];
				}
			}
		}
		return dst;
	}

	outW = width;
	outH = height;
	return std::vector<uint8_t>(src, src + static_cast<size_t>(width) * static_cast<size_t>(height));
}

static std::vector<uint8_t> addWhiteBorder(const uint8_t* src, int width, int height, int pad, int& outW, int& outH)
{
	if (pad <= 0) {
		outW = width;
		outH = height;
		return std::vector<uint8_t>(src, src + static_cast<size_t>(width) * static_cast<size_t>(height));
	}
	outW = width + pad * 2;
	outH = height + pad * 2;
	std::vector<uint8_t> dst(static_cast<size_t>(outW) * static_cast<size_t>(outH), 255);
	for (int y = 0; y < height; y++) {
		auto* drow = dst.data() + static_cast<size_t>(y + pad) * outW + pad;
		const auto* srow = src + static_cast<size_t>(y) * width;
		std::copy(srow, srow + width, drow);
	}
	return dst;
}

struct Margins {
	int top = 0;
	int bottom = 0;
	int left = 0;
	int right = 0;
};

static bool isMostlyWhiteRow(const uint8_t* row, int width)
{
	int dark = 0;
	int sum = 0;
	uint64_t sumSq = 0;
	for (int x = 0; x < width; x += 2) {
		const int v = row[x];
		sum += v;
		sumSq += static_cast<uint64_t>(v) * static_cast<uint64_t>(v);
		if (v < 180)
			dark++;
	}
	const int n = (width + 1) / 2;
	const double mean = static_cast<double>(sum) / std::max(1, n);
	const double meanSq = static_cast<double>(sumSq) / std::max(1, n);
	const double var = std::max(0.0, meanSq - mean * mean);
	const double std = std::sqrt(var);
	const double darkRatio = static_cast<double>(dark) / std::max(1, n);
	if (mean >= 205.0 && std <= 22.0 && darkRatio <= 0.10)
		return true;
	if (mean >= 195.0 && std <= 32.0 && darkRatio <= 0.12)
		return true;
	return mean >= 185.0 && std <= 42.0 && darkRatio <= 0.08;
}

static bool isMostlyWhiteCol(const uint8_t* src, int width, int height, int x)
{
	int dark = 0;
	int sum = 0;
	uint64_t sumSq = 0;
	for (int y = 0; y < height; y += 2) {
		const int v = src[y * width + x];
		sum += v;
		sumSq += static_cast<uint64_t>(v) * static_cast<uint64_t>(v);
		if (v < 180)
			dark++;
	}
	const int n = (height + 1) / 2;
	const double mean = static_cast<double>(sum) / std::max(1, n);
	const double meanSq = static_cast<double>(sumSq) / std::max(1, n);
	const double var = std::max(0.0, meanSq - mean * mean);
	const double std = std::sqrt(var);
	const double darkRatio = static_cast<double>(dark) / std::max(1, n);
	if (mean >= 205.0 && std <= 22.0 && darkRatio <= 0.10)
		return true;
	if (mean >= 195.0 && std <= 32.0 && darkRatio <= 0.12)
		return true;
	return mean >= 185.0 && std <= 42.0 && darkRatio <= 0.08;
}

static Margins measureWhiteMargins(const uint8_t* src, int width, int height)
{
	Margins m{};
	for (int y = 0; y < height; y++) {
		if (isMostlyWhiteRow(src + y * width, width)) {
			m.top++;
		} else {
			break;
		}
	}
	for (int y = height - 1; y >= 0; y--) {
		if (isMostlyWhiteRow(src + y * width, width)) {
			m.bottom++;
		} else {
			break;
		}
	}

	for (int x = 0; x < width; x += 2) {
		if (isMostlyWhiteCol(src, width, height, x)) {
			m.left += 2;
		} else {
			break;
		}
	}
	m.left = std::min(m.left, width);

	for (int x = width - 1; x >= 0; x -= 2) {
		if (isMostlyWhiteCol(src, width, height, x)) {
			m.right += 2;
		} else {
			break;
		}
	}
	m.right = std::min(m.right, width);
	return m;
}

static std::vector<uint8_t> addWhiteBorderPerSide(
	const uint8_t* src, int width, int height,
	int padLeft, int padTop, int padRight, int padBottom,
	int& outW, int& outH)
{
	padLeft = std::max(0, padLeft);
	padTop = std::max(0, padTop);
	padRight = std::max(0, padRight);
	padBottom = std::max(0, padBottom);
	outW = width + padLeft + padRight;
	outH = height + padTop + padBottom;
	std::vector<uint8_t> dst(static_cast<size_t>(outW) * static_cast<size_t>(outH), 255);
	for (int y = 0; y < height; y++) {
		auto* drow = dst.data() + static_cast<size_t>(y + padTop) * outW + padLeft;
		const auto* srow = src + static_cast<size_t>(y) * width;
		std::copy(srow, srow + width, drow);
	}
	return dst;
}

static std::vector<uint8_t> resizeGrayBilinear(const uint8_t* src, int srcW, int srcH, int dstW, int dstH)
{
	dstW = std::max(1, dstW);
	dstH = std::max(1, dstH);
	std::vector<uint8_t> dst(static_cast<size_t>(dstW) * static_cast<size_t>(dstH));
	if (srcW <= 1 || srcH <= 1 || dstW == 1 || dstH == 1) {
		for (int y = 0; y < dstH; y++) {
			const int sy = (srcH <= 1) ? 0 : (y * (srcH - 1)) / std::max(1, dstH - 1);
			for (int x = 0; x < dstW; x++) {
				const int sx = (srcW <= 1) ? 0 : (x * (srcW - 1)) / std::max(1, dstW - 1);
				dst[static_cast<size_t>(y) * dstW + x] = src[static_cast<size_t>(sy) * srcW + sx];
			}
		}
		return dst;
	}

	const double xScale = static_cast<double>(srcW - 1) / static_cast<double>(std::max(1, dstW - 1));
	const double yScale = static_cast<double>(srcH - 1) / static_cast<double>(std::max(1, dstH - 1));
	for (int y = 0; y < dstH; y++) {
		const double fy = y * yScale;
		const int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, srcH - 1);
		const int y1 = std::min(srcH - 1, y0 + 1);
		const double wy = fy - y0;
		for (int x = 0; x < dstW; x++) {
			const double fx = x * xScale;
			const int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, srcW - 1);
			const int x1 = std::min(srcW - 1, x0 + 1);
			const double wx = fx - x0;

			const int p00 = src[static_cast<size_t>(y0) * srcW + x0];
			const int p10 = src[static_cast<size_t>(y0) * srcW + x1];
			const int p01 = src[static_cast<size_t>(y1) * srcW + x0];
			const int p11 = src[static_cast<size_t>(y1) * srcW + x1];

			const double a = p00 + (p10 - p00) * wx;
			const double b = p01 + (p11 - p01) * wx;
			const int v = static_cast<int>(std::lround(a + (b - a) * wy));
			dst[static_cast<size_t>(y) * dstW + x] = static_cast<uint8_t>(std::clamp(v, 0, 255));
		}
	}
	return dst;
}

static std::string formatRect(RectI r)
{
	return std::to_string(r.left) + "," + std::to_string(r.top) + "," + std::to_string(r.right) + "," + std::to_string(r.bottom);
}

static std::string formatQuad(int x0, int y0, int x1, int y1, int x2, int y2, int x3, int y3)
{
	return std::to_string(x0) + "," + std::to_string(y0) + "," + std::to_string(x1) + "," + std::to_string(y1) + "," +
		   std::to_string(x2) + "," + std::to_string(y2) + "," + std::to_string(x3) + "," + std::to_string(y3);
}

static RectI rectFromPosition(const ZXing::Position& pos, RectI roiOffset)
{
	int minX = std::min({pos.topLeft().x, pos.topRight().x, pos.bottomLeft().x, pos.bottomRight().x});
	int maxX = std::max({pos.topLeft().x, pos.topRight().x, pos.bottomLeft().x, pos.bottomRight().x});
	int minY = std::min({pos.topLeft().y, pos.topRight().y, pos.bottomLeft().y, pos.bottomRight().y});
	int maxY = std::max({pos.topLeft().y, pos.topRight().y, pos.bottomLeft().y, pos.bottomRight().y});
	return RectI{roiOffset.left + minX, roiOffset.top + minY, roiOffset.left + maxX, roiOffset.top + maxY};
}

static void mapPointFromRotToCrop(int rot, int cropW, int cropH, int xRot, int yRot, int& outX, int& outY)
{
	if (rot == 90) {
		outX = yRot;
		outY = cropH - 1 - xRot;
	} else if (rot == 270) {
		outX = cropW - 1 - yRot;
		outY = xRot;
	} else if (rot == 180) {
		outX = cropW - 1 - xRot;
		outY = cropH - 1 - yRot;
	} else {
		outX = xRot;
		outY = yRot;
	}
}

static RectI mapRectFromRotToCrop(int rot, int cropW, int cropH, RectI rInRot)
{
	int xs[4] = {rInRot.left, rInRot.right, rInRot.right, rInRot.left};
	int ys[4] = {rInRot.top, rInRot.top, rInRot.bottom, rInRot.bottom};
	int minX = std::numeric_limits<int>::max();
	int minY = std::numeric_limits<int>::max();
	int maxX = std::numeric_limits<int>::min();
	int maxY = std::numeric_limits<int>::min();
	for (int i = 0; i < 4; i++) {
		int xC = 0, yC = 0;
		mapPointFromRotToCrop(rot, cropW, cropH, xs[i], ys[i], xC, yC);
		minX = std::min(minX, xC);
		minY = std::min(minY, yC);
		maxX = std::max(maxX, xC);
		maxY = std::max(maxY, yC);
	}
	return clampRect(RectI{minX, minY, maxX, maxY}, cropW, cropH);
}

static bool looksSyntheticQR(const uint8_t* src, int width, int height)
{
	if (!src || width <= 0 || height <= 0)
		return false;
	const int stepX = std::max(1, width / 64);
	const int stepY = std::max(1, height / 64);
	int n = 0;
	int hits = 0;
	for (int y = 0; y < height; y += stepY) {
		const auto* row = src + static_cast<size_t>(y) * width;
		for (int x = 0; x < width; x += stepX) {
			const uint8_t v = row[x];
			if (v == 0 || v == 255 || v == 127)
				hits++;
			n++;
		}
	}
	return n > 0 && static_cast<double>(hits) / static_cast<double>(n) >= 0.985;
}

static std::vector<uint8_t> mapUnknownValue(const uint8_t* src, int width, int height, uint8_t unknownValue, uint8_t mappedTo)
{
	std::vector<uint8_t> out(static_cast<size_t>(width) * static_cast<size_t>(height));
	const size_t n = out.size();
	for (size_t i = 0; i < n; i++) {
		const uint8_t v = src[i];
		out[i] = (v == unknownValue) ? mappedTo : v;
	}
	return out;
}

static std::string trimPipes(std::string s)
{
	for (auto& c : s) {
		if (c == '|')
			c = '/';
	}
	return s;
}

static std::string hexPrefix(const ZXing::ByteArray& bytes, size_t maxBytes)
{
	static const char* hex = "0123456789ABCDEF";
	const size_t n = std::min(maxBytes, bytes.size());
	std::string out;
	out.reserve(n * 2);
	for (size_t i = 0; i < n; i++) {
		const uint8_t b = bytes[i];
		out.push_back(hex[(b >> 4) & 0xF]);
		out.push_back(hex[b & 0xF]);
	}
	return out;
}

static std::string errorTypeToStr(ZXing::Error::Type t)
{
	switch (t) {
	case ZXing::Error::Type::Format:
		return "Format";
	case ZXing::Error::Type::Checksum:
		return "Checksum";
	case ZXing::Error::Type::Unsupported:
		return "Unsupported";
	default:
		return "None";
	}
}

static std::string buildInvalidDiagnosticText(const ZXing::Barcode& barcode, bool includeText)
{
	std::string text = includeText ? trimPipes(barcode.text()) : std::string();
	if (!text.empty()) {
		if (text.size() > 96)
			text.resize(96);
	}

	const auto fmt = trimPipes(ZXing::ToString(barcode.format()));
	const auto ver = trimPipes(barcode.version());
	const auto ecl = trimPipes(barcode.ecLevel());
	const auto si = trimPipes(barcode.symbologyIdentifier());
	const auto err = barcode.error();
	const auto errType = errorTypeToStr(err.type());
	const auto errMsg = trimPipes(err.msg().substr(0, std::min<size_t>(64, err.msg().size())));

	const auto bytes = barcode.bytes();
	const auto hex0 = hexPrefix(bytes, 28);

	std::string hexEci0;
	if (barcode.hasECI()) {
		const auto eciBytes = barcode.bytesECI();
		hexEci0 = hexPrefix(eciBytes, 28);
	}

	std::string out = "INVALID(" + errType + (barcode.isValid() ? ",V" : ",I") + ")";
	out += " fmt=" + fmt;
	if (!ver.empty())
		out += " ver=" + ver;
	if (!ecl.empty())
		out += " ecl=" + ecl;
	if (!si.empty())
		out += " si=" + si;
	out += " mir=" + std::to_string(barcode.isMirrored() ? 1 : 0);
	out += " inv=" + std::to_string(barcode.isInverted() ? 1 : 0);
	out += " ori=" + std::to_string(barcode.orientation());
	out += " len=" + std::to_string(bytes.size());
	if (!hex0.empty())
		out += " hex=" + hex0;
	if (!hexEci0.empty())
		out += " eciHex=" + hexEci0;
	if (!errMsg.empty())
		out += " msg=" + errMsg;
	if (!text.empty())
		out += " txt=" + text;
	return out;
}

static ZXing::Barcode retryAsQRCodeIfMicroInvalid(const ZXing::ImageView& image, const ZXing::ReaderOptions& opts, const ZXing::Barcode& current)
{
	if (current.format() != ZXing::BarcodeFormat::MicroQRCode)
		return current;
	if (current.isValid())
		return current;
	if (current.error().type() != ZXing::Error::Type::Format)
		return current;

	ZXing::ReaderOptions opts2 = opts;
	opts2.setFormats(ZXing::BarcodeFormat::QRCode);
	auto retry = ZXing::ReadBarcode(image, opts2);
	if (retry.format() != ZXing::BarcodeFormat::None && (retry.isValid() || !retry.text().empty()))
		return retry;

	return current;
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_smartqrcode_NativeBridge_hello(JNIEnv* env, jobject /*thiz*/)
{
	return env->NewStringUTF("native-ok");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_smartqrcode_NativeDecoder_decodeGray(JNIEnv* env,
											 jobject /*thiz*/,
											 jbyteArray gray,
											 jint width,
											 jint height,
											 jint rotationDegrees,
											 jint roiLeft,
											 jint roiTop,
											 jint roiRight,
											 jint roiBottom,
											 jboolean tryHarder)
{
	if (gray == nullptr)
		return nullptr;

	const int w = static_cast<int>(width);
	const int h = static_cast<int>(height);
	if (w <= 0 || h <= 0)
		return nullptr;

	const jsize len = env->GetArrayLength(gray);
	const size_t expected = static_cast<size_t>(w) * static_cast<size_t>(h);
	if (len < 0 || static_cast<size_t>(len) < expected)
		return nullptr;

	jboolean isCopy = JNI_FALSE;
	auto* data = reinterpret_cast<uint8_t*>(env->GetByteArrayElements(gray, &isCopy));
	if (!data)
		return nullptr;

	RectI roi{static_cast<int>(roiLeft), static_cast<int>(roiTop), static_cast<int>(roiRight), static_cast<int>(roiBottom)};
	roi = clampRect(roi, w, h);
	if (roi.right == roi.left || roi.bottom == roi.top) {
		env->ReleaseByteArrayElements(gray, reinterpret_cast<jbyte*>(data), JNI_ABORT);
		return nullptr;
	}

	int cropW = 0;
	int cropH = 0;
	std::vector<uint8_t> cropped = cropGray(data, w, h, roi, cropW, cropH);
	env->ReleaseByteArrayElements(gray, reinterpret_cast<jbyte*>(data), JNI_ABORT);

	int rotW = 0;
	int rotH = 0;
	std::vector<uint8_t> rotated = rotateGray(cropped.data(), cropW, cropH, static_cast<int>(rotationDegrees), rotW, rotH);

	const bool fullRoi = roi.left == 0 && roi.top == 0 && roi.right == w && roi.bottom == h;
	const int rot = (((rotationDegrees % 360) + 360) % 360);
	if (fullRoi && rot == 0 && rotW == rotH && rotW <= 900 && looksSyntheticQR(rotated.data(), rotW, rotH)) {
		auto tryPure = [&](const uint8_t* buf, int bufW, int bufH, bool th, ZXing::Binarizer bin) -> ZXing::Barcode {
			auto image = ZXing::ImageView(buf, bufW, bufH, ZXing::ImageFormat::Lum);
			ZXing::ReaderOptions opts;
			opts.setFormats(ZXing::BarcodeFormat::QRCode | ZXing::BarcodeFormat::MicroQRCode);
			opts.setIsPure(true);
			opts.setTryHarder(th);
			opts.setTryRotate(false);
			opts.setTryDownscale(false);
			opts.setTryInvert(th);
			opts.setReturnErrors(true);
			opts.setBinarizer(bin);
			auto b = ZXing::ReadBarcode(image, opts);
			return retryAsQRCodeIfMicroInvalid(image, opts, b);
		};

		std::vector<uint8_t> unknownToWhite = mapUnknownValue(rotated.data(), rotW, rotH, 127, 255);
		std::vector<uint8_t> unknownToBlack = mapUnknownValue(rotated.data(), rotW, rotH, 127, 0);

		ZXing::Barcode b;
		const bool th = (tryHarder == JNI_TRUE);
		const int pads[] = {0, 16, 32};
		for (int pad : pads) {
			int pw = 0;
			int ph = 0;
			std::vector<uint8_t> padded = addWhiteBorder(rotated.data(), rotW, rotH, pad, pw, ph);
			std::vector<uint8_t> paddedW = addWhiteBorder(unknownToWhite.data(), rotW, rotH, pad, pw, ph);
			std::vector<uint8_t> paddedB = addWhiteBorder(unknownToBlack.data(), rotW, rotH, pad, pw, ph);

			const uint8_t* bufs[] = {padded.data(), paddedW.data(), paddedB.data()};
			const int bufWs[] = {pw, pw, pw};
			const int bufHs[] = {ph, ph, ph};
			for (int i = 0; i < 3; i++) {
				b = tryPure(bufs[i], bufWs[i], bufHs[i], th, ZXing::Binarizer::FixedThreshold);
				if (b.format() == ZXing::BarcodeFormat::None)
					b = tryPure(bufs[i], bufWs[i], bufHs[i], th, ZXing::Binarizer::LocalAverage);
				if (b.format() != ZXing::BarcodeFormat::None) {
					auto text = b.text();
					if (!b.isValid() || text.empty())
						text = buildInvalidDiagnosticText(b, true);

					const auto& pos = b.position();
					const int tlx = std::clamp(pos.topLeft().x - pad, 0, std::max(0, rotW - 1));
					const int tly = std::clamp(pos.topLeft().y - pad, 0, std::max(0, rotH - 1));
					const int trx = std::clamp(pos.topRight().x - pad, 0, std::max(0, rotW - 1));
					const int tryy = std::clamp(pos.topRight().y - pad, 0, std::max(0, rotH - 1));
					const int brx = std::clamp(pos.bottomRight().x - pad, 0, std::max(0, rotW - 1));
					const int bry = std::clamp(pos.bottomRight().y - pad, 0, std::max(0, rotH - 1));
					const int blx = std::clamp(pos.bottomLeft().x - pad, 0, std::max(0, rotW - 1));
					const int bly = std::clamp(pos.bottomLeft().y - pad, 0, std::max(0, rotH - 1));

					const int minX = std::min({tlx, trx, brx, blx});
					const int maxX = std::max({tlx, trx, brx, blx});
					const int minY = std::min({tly, tryy, bry, bly});
					const int maxY = std::max({tly, tryy, bry, bly});

					const std::string out = text + "|" + formatRect(RectI{minX, minY, maxX, maxY}) + "|" + formatQuad(tlx, tly, trx, tryy, brx, bry, blx, bly);
					return env->NewStringUTF(out.c_str());
				}
			}
		}
		if (tryHarder == JNI_TRUE)
			return nullptr;
	}

	const int minDim = std::max(1, std::min(rotW, rotH));
	Margins margins = measureWhiteMargins(rotated.data(), rotW, rotH);
	const bool qzSuspect = margins.left <= 2 || margins.right <= 2 || margins.top <= 2 || margins.bottom <= 2;
	const bool qzZero = margins.left == 0 && margins.right == 0 && margins.top == 0 && margins.bottom == 0;
	int targetQzBase = minDim / 10;
	if (tryHarder == JNI_TRUE) {
		targetQzBase = std::max(targetQzBase, minDim / 7);
		if (qzSuspect)
			targetQzBase = std::max(targetQzBase, minDim / 6);
	} else {
		if (qzSuspect)
			targetQzBase = std::max(targetQzBase, minDim / 8);
	}
	if (qzZero)
		targetQzBase = std::max(targetQzBase, (tryHarder == JNI_TRUE) ? (minDim / 4) : (minDim / 5));
	const int targetQzMax = (tryHarder == JNI_TRUE) ? (qzZero ? 220 : (qzSuspect ? 140 : 96)) : (qzZero ? 160 : (qzSuspect ? 96 : 64));
	const int targetQz = std::clamp(targetQzBase, (tryHarder == JNI_TRUE) ? 20 : 14, targetQzMax);
	const int padL = std::max(0, targetQz - margins.left);
	const int padR = std::max(0, targetQz - margins.right);
	const int padT = std::max(0, targetQz - margins.top);
	const int padB = std::max(0, targetQz - margins.bottom);
	int baseW = 0;
	int baseH = 0;
	std::vector<uint8_t> base = addWhiteBorderPerSide(rotated.data(), rotW, rotH, padL, padT, padR, padB, baseW, baseH);
	const int padCap = qzZero ? 512 : ((std::max(baseW, baseH) <= 900) ? 192 : 512);
	const int basePadMax = (tryHarder == JNI_TRUE) ? (qzZero ? padCap : (qzSuspect ? 320 : 192)) : (qzZero ? 256 : (qzSuspect ? 192 : 128));
	const int basePadMin = (qzZero ? ((tryHarder == JNI_TRUE) ? 128 : 96) : (qzSuspect ? 32 : 16));
	const int basePadBasis = qzZero ? std::max(targetQz, minDim / 6) : std::max(targetQz, minDim / 8);
	const int basePad = std::clamp(basePadBasis, basePadMin, basePadMax);
	std::vector<int> pads = {
		0,
		std::clamp(basePad / 2, 0, 96),
		basePad,
	};
	if (tryHarder == JNI_TRUE) {
		if (qzZero) {
			pads.push_back(std::clamp(basePad * 2, basePad + 1, padCap));
			pads.push_back(std::clamp(basePad * 3, basePad * 2 + 1, padCap));
			pads.push_back(std::clamp(basePad * 4, basePad * 3 + 1, padCap));
			pads.push_back(std::clamp(basePad * 6, basePad * 4 + 1, padCap));
		} else {
			pads.push_back(std::clamp(basePad * 2, basePad + 1, 192));
			pads.push_back(std::clamp(basePad * 3, basePad * 2 + 1, 256));
			pads.push_back(std::clamp(basePad / 3, 0, 64));
			if (qzSuspect) {
				pads.push_back(std::clamp(basePad * 4, basePad * 3 + 1, padCap));
				pads.push_back(std::clamp(basePad * 6, basePad * 4 + 1, padCap));
			}
		}
	}
	if (qzZero) {
		std::sort(pads.begin(), pads.end(), std::greater<int>());
	} else {
		std::sort(pads.begin(), pads.end());
	}
	pads.erase(std::unique(pads.begin(), pads.end()), pads.end());

	ZXing::Barcode barcode;
	ZXing::Barcode bestInvalid;
	bool haveInvalid = false;
	int padUsed = 0;
	int bestInvalidPad = 0;
	double scaleUsed = 1.0;
	double bestInvalidScale = 1.0;
	for (int pad : pads) {
		int paddedW = 0;
		int paddedH = 0;
		std::vector<uint8_t> padded = addWhiteBorder(base.data(), baseW, baseH, pad, paddedW, paddedH);
		const int maxSide = std::max(paddedW, paddedH);
		std::vector<double> scales = {1.0};
		const bool th = (tryHarder == JNI_TRUE);
		if (th && std::max(paddedW, paddedH) >= 1100)
			scales.push_back(0.75);
		if (th) {
			if (qzZero) {
				scales.push_back(0.5);
				scales.push_back(0.66);
				scales.push_back(0.75);
				scales.push_back(1.25);
			}
			if (qzZero && maxSide >= 900) {
				scales.push_back(1.5);
				scales.push_back(2.0);
				scales.push_back(2.5);
			} else if (qzSuspect && maxSide >= 900) {
				scales.push_back(0.66);
				scales.push_back(1.5);
			}
		}
		if (qzZero) {
			std::sort(scales.begin(), scales.end(), std::greater<double>());
		} else {
			std::sort(scales.begin(), scales.end());
		}
		scales.erase(std::unique(scales.begin(), scales.end()), scales.end());

		bool decoded = false;
		for (double scale : scales) {
			const uint8_t* buf = padded.data();
			int bufW = paddedW;
			int bufH = paddedH;
			std::vector<uint8_t> resized;
			if (scale != 1.0) {
				const int dstW = std::clamp(static_cast<int>(std::lround(paddedW * scale)), 32, 2200);
				const int dstH = std::clamp(static_cast<int>(std::lround(paddedH * scale)), 32, 2200);
				resized = resizeGrayBilinear(padded.data(), paddedW, paddedH, dstW, dstH);
				buf = resized.data();
				bufW = dstW;
				bufH = dstH;
			}

			auto image = ZXing::ImageView(buf, bufW, bufH, ZXing::ImageFormat::Lum);

			ZXing::ReaderOptions opts;
			opts.setFormats(ZXing::BarcodeFormat::QRCode | ZXing::BarcodeFormat::MicroQRCode);
			const bool th = (tryHarder == JNI_TRUE);
			opts.setTryHarder(th);
			opts.setTryInvert(th);
			opts.setTryRotate(false);
			opts.setTryDownscale(th);
			opts.setReturnErrors(th);
#ifdef ZXING_EXPERIMENTAL_API
			if (th)
				opts.setTryDenoise(true);
#endif
			barcode = ZXing::ReadBarcode(image, opts);
			barcode = retryAsQRCodeIfMicroInvalid(image, opts, barcode);
			if (barcode.format() != ZXing::BarcodeFormat::None && !barcode.isValid() && !haveInvalid) {
				bestInvalid = barcode;
				haveInvalid = true;
				bestInvalidPad = pad;
				bestInvalidScale = scale;
			}
			if (barcode.error() == ZXing::Error::Format && tryHarder != JNI_TRUE)
				barcode = ZXing::Barcode();
			if (barcode.format() == ZXing::BarcodeFormat::None && tryHarder == JNI_TRUE) {
				opts.setBinarizer(ZXing::Binarizer::GlobalHistogram);
				barcode = ZXing::ReadBarcode(image, opts);
				barcode = retryAsQRCodeIfMicroInvalid(image, opts, barcode);
				if (barcode.format() != ZXing::BarcodeFormat::None && !barcode.isValid() && !haveInvalid) {
					bestInvalid = barcode;
					haveInvalid = true;
					bestInvalidPad = pad;
					bestInvalidScale = scale;
				}
				if (barcode.error() == ZXing::Error::Format)
					barcode = ZXing::Barcode();
			}
			if (barcode.format() == ZXing::BarcodeFormat::None && tryHarder == JNI_TRUE) {
				opts.setBinarizer(ZXing::Binarizer::FixedThreshold);
				barcode = ZXing::ReadBarcode(image, opts);
				barcode = retryAsQRCodeIfMicroInvalid(image, opts, barcode);
				if (barcode.format() != ZXing::BarcodeFormat::None && !barcode.isValid() && !haveInvalid) {
					bestInvalid = barcode;
					haveInvalid = true;
					bestInvalidPad = pad;
					bestInvalidScale = scale;
				}
				if (barcode.error() == ZXing::Error::Format)
					barcode = ZXing::Barcode();
			}
			if (barcode.format() == ZXing::BarcodeFormat::None && tryHarder == JNI_TRUE) {
				opts.setTryRotate(true);
				opts.setBinarizer(ZXing::Binarizer::LocalAverage);
				barcode = ZXing::ReadBarcode(image, opts);
				barcode = retryAsQRCodeIfMicroInvalid(image, opts, barcode);
				if (barcode.format() != ZXing::BarcodeFormat::None && !barcode.isValid() && !haveInvalid) {
					bestInvalid = barcode;
					haveInvalid = true;
					bestInvalidPad = pad;
					bestInvalidScale = scale;
				}
				if (barcode.error() == ZXing::Error::Format)
					barcode = ZXing::Barcode();
			}
			if (barcode.format() == ZXing::BarcodeFormat::None && tryHarder == JNI_TRUE) {
				opts.setTryRotate(true);
				opts.setBinarizer(ZXing::Binarizer::FixedThreshold);
				barcode = ZXing::ReadBarcode(image, opts);
				barcode = retryAsQRCodeIfMicroInvalid(image, opts, barcode);
				if (barcode.format() != ZXing::BarcodeFormat::None && !barcode.isValid() && !haveInvalid) {
					bestInvalid = barcode;
					haveInvalid = true;
					bestInvalidPad = pad;
					bestInvalidScale = scale;
				}
				if (barcode.error() == ZXing::Error::Format)
					barcode = ZXing::Barcode();
			}
			if (barcode.format() != ZXing::BarcodeFormat::None) {
				padUsed = pad;
				scaleUsed = scale;
				decoded = true;
				break;
			}
		}
		if (decoded)
			break;
	}
	if (barcode.format() == ZXing::BarcodeFormat::None && haveInvalid) {
		barcode = bestInvalid;
		padUsed = bestInvalidPad;
		scaleUsed = bestInvalidScale;
	}
	if (barcode.format() == ZXing::BarcodeFormat::None) {
		if (tryHarder == JNI_TRUE) {
			std::string padsStr;
			for (size_t i = 0; i < pads.size(); i++) {
				padsStr += std::to_string(pads[i]);
				if (i + 1 < pads.size())
					padsStr += ",";
			}
			__android_log_print(
				ANDROID_LOG_INFO,
				"SmartQRCode",
				"NativeFail rotDeg=%d roi=%s crop=%dx%d rot=%dx%d qz=%d,%d,%d,%d qz0=%d tqz=%d pside=%d,%d,%d,%d base=%dx%d bpad=%d pads=%s",
				static_cast<int>(rotationDegrees),
				formatRect(roi).c_str(),
				cropW,
				cropH,
				rotW,
				rotH,
				margins.left,
				margins.top,
				margins.right,
				margins.bottom,
				qzZero ? 1 : 0,
				targetQz,
				padL,
				padT,
				padR,
				padB,
				baseW,
				baseH,
				basePad,
				padsStr.c_str()
			);
		}
		return nullptr;
	}

	auto text = barcode.text();
	if (tryHarder == JNI_TRUE && text.empty()) {
		__android_log_print(
			ANDROID_LOG_INFO,
			"SmartQRCode",
			"NativeHintEmpty rotDeg=%d roi=%s crop=%dx%d rot=%dx%d",
			static_cast<int>(rotationDegrees),
			formatRect(roi).c_str(),
			cropW,
			cropH,
			rotW,
			rotH
		);
	}
	if (tryHarder == JNI_TRUE && (!barcode.isValid() || text.empty())) {
		text = buildInvalidDiagnosticText(barcode, true);
	}

	const auto& pos = barcode.position();
	const int offX = padUsed + padL;
	const int offY = padUsed + padT;
	const double invScale = (scaleUsed != 0.0) ? (1.0 / scaleUsed) : 1.0;
	const int pTlx = static_cast<int>(std::lround(pos.topLeft().x * invScale));
	const int pTly = static_cast<int>(std::lround(pos.topLeft().y * invScale));
	const int pTrx = static_cast<int>(std::lround(pos.topRight().x * invScale));
	const int pTry = static_cast<int>(std::lround(pos.topRight().y * invScale));
	const int pBrx = static_cast<int>(std::lround(pos.bottomRight().x * invScale));
	const int pBry = static_cast<int>(std::lround(pos.bottomRight().y * invScale));
	const int pBlx = static_cast<int>(std::lround(pos.bottomLeft().x * invScale));
	const int pBly = static_cast<int>(std::lround(pos.bottomLeft().y * invScale));

	const int tlxR = std::clamp(pTlx - offX, 0, std::max(0, rotW - 1));
	const int tlyR = std::clamp(pTly - offY, 0, std::max(0, rotH - 1));
	const int trxR = std::clamp(pTrx - offX, 0, std::max(0, rotW - 1));
	const int tryR = std::clamp(pTry - offY, 0, std::max(0, rotH - 1));
	const int brxR = std::clamp(pBrx - offX, 0, std::max(0, rotW - 1));
	const int bryR = std::clamp(pBry - offY, 0, std::max(0, rotH - 1));
	const int blxR = std::clamp(pBlx - offX, 0, std::max(0, rotW - 1));
	const int blyR = std::clamp(pBly - offY, 0, std::max(0, rotH - 1));

	const int minX = std::min({tlxR, trxR, brxR, blxR});
	const int maxX = std::max({tlxR, trxR, brxR, blxR});
	const int minY = std::min({tlyR, tryR, bryR, blyR});
	const int maxY = std::max({tlyR, tryR, bryR, blyR});
	RectI boxInRot = RectI{minX, minY, maxX, maxY};
	RectI boxInCrop = mapRectFromRotToCrop(rot, cropW, cropH, boxInRot);

	RectI boxInFull = RectI{roi.left + boxInCrop.left, roi.top + boxInCrop.top, roi.left + boxInCrop.right, roi.top + boxInCrop.bottom};

	int tlxC = 0, tlyC = 0, trxC = 0, tryC = 0, brxC = 0, bryC = 0, blxC = 0, blyC = 0;
	mapPointFromRotToCrop(rot, cropW, cropH, tlxR, tlyR, tlxC, tlyC);
	mapPointFromRotToCrop(rot, cropW, cropH, trxR, tryR, trxC, tryC);
	mapPointFromRotToCrop(rot, cropW, cropH, brxR, bryR, brxC, bryC);
	mapPointFromRotToCrop(rot, cropW, cropH, blxR, blyR, blxC, blyC);
	const std::string quadInFull = formatQuad(
		roi.left + tlxC, roi.top + tlyC,
		roi.left + trxC, roi.top + tryC,
		roi.left + brxC, roi.top + bryC,
		roi.left + blxC, roi.top + blyC
	);

	const std::string out = text + "|" + formatRect(boxInFull) + "|" + quadInFull;
	return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_smartqrcode_NativeDecoder_decodeGrayDebug(JNIEnv* env,
												  jobject /*thiz*/,
												  jbyteArray gray,
												  jint width,
												  jint height,
												  jint rotationDegrees,
												  jint roiLeft,
												  jint roiTop,
												  jint roiRight,
												  jint roiBottom,
												  jboolean tryHarder)
{
	if (gray == nullptr)
		return env->NewStringUTF("null-gray");

	const int w = static_cast<int>(width);
	const int h = static_cast<int>(height);
	if (w <= 0 || h <= 0)
		return env->NewStringUTF("bad-size");

	const jsize len = env->GetArrayLength(gray);
	const size_t expected = static_cast<size_t>(w) * static_cast<size_t>(h);
	if (len < 0 || static_cast<size_t>(len) < expected)
		return env->NewStringUTF("bad-len");

	jboolean isCopy = JNI_FALSE;
	auto* data = reinterpret_cast<uint8_t*>(env->GetByteArrayElements(gray, &isCopy));
	if (!data)
		return env->NewStringUTF("no-bytes");

	RectI roi{static_cast<int>(roiLeft), static_cast<int>(roiTop), static_cast<int>(roiRight), static_cast<int>(roiBottom)};
	roi = clampRect(roi, w, h);
	if (roi.right == roi.left || roi.bottom == roi.top) {
		env->ReleaseByteArrayElements(gray, reinterpret_cast<jbyte*>(data), JNI_ABORT);
		return env->NewStringUTF("empty-roi");
	}

	int cropW = 0;
	int cropH = 0;
	std::vector<uint8_t> cropped = cropGray(data, w, h, roi, cropW, cropH);
	env->ReleaseByteArrayElements(gray, reinterpret_cast<jbyte*>(data), JNI_ABORT);

	int rotW = 0;
	int rotH = 0;
	std::vector<uint8_t> rotated = rotateGray(cropped.data(), cropW, cropH, static_cast<int>(rotationDegrees), rotW, rotH);

	const int minDim = std::max(1, std::min(rotW, rotH));
	Margins margins = measureWhiteMargins(rotated.data(), rotW, rotH);
	const bool qzSuspect = margins.left <= 2 || margins.right <= 2 || margins.top <= 2 || margins.bottom <= 2;
	const bool qzZero = margins.left == 0 && margins.right == 0 && margins.top == 0 && margins.bottom == 0;
	int targetQzBase = minDim / 10;
	if (tryHarder == JNI_TRUE) {
		targetQzBase = std::max(targetQzBase, minDim / 7);
		if (qzSuspect)
			targetQzBase = std::max(targetQzBase, minDim / 6);
	} else {
		if (qzSuspect)
			targetQzBase = std::max(targetQzBase, minDim / 8);
	}
	if (qzZero)
		targetQzBase = std::max(targetQzBase, minDim / 5);
	const int targetQzMax = (tryHarder == JNI_TRUE) ? (qzZero ? 220 : (qzSuspect ? 140 : 96)) : (qzZero ? 160 : (qzSuspect ? 96 : 64));
	const int targetQz = std::clamp(targetQzBase, (tryHarder == JNI_TRUE) ? 20 : 14, targetQzMax);
	const int padL = std::max(0, targetQz - margins.left);
	const int padR = std::max(0, targetQz - margins.right);
	const int padT = std::max(0, targetQz - margins.top);
	const int padB = std::max(0, targetQz - margins.bottom);
	int baseW = 0;
	int baseH = 0;
	std::vector<uint8_t> base = addWhiteBorderPerSide(rotated.data(), rotW, rotH, padL, padT, padR, padB, baseW, baseH);
	const int padCap = qzZero ? 512 : ((std::max(baseW, baseH) <= 900) ? 192 : 512);
	const int basePadMax = (tryHarder == JNI_TRUE) ? (qzZero ? padCap : (qzSuspect ? 320 : 192)) : (qzZero ? 256 : (qzSuspect ? 192 : 128));
	const int basePadMin = (qzZero ? 96 : (qzSuspect ? 32 : 16));
	const int basePadBasis = qzZero ? std::max(targetQz, minDim / 6) : std::max(targetQz, minDim / 8);
	const int basePad = std::clamp(basePadBasis, basePadMin, basePadMax);
	std::vector<int> pads = {
		0,
		std::clamp(basePad / 2, 0, 96),
		basePad,
	};
	if (tryHarder == JNI_TRUE) {
		if (qzZero) {
			pads.push_back(std::clamp(basePad * 2, basePad + 1, padCap));
			pads.push_back(std::clamp(basePad * 3, basePad * 2 + 1, padCap));
			pads.push_back(std::clamp(basePad * 4, basePad * 3 + 1, padCap));
		} else {
			pads.push_back(std::clamp(basePad * 2, basePad + 1, 192));
			pads.push_back(std::clamp(basePad * 3, basePad * 2 + 1, 256));
			pads.push_back(std::clamp(basePad / 3, 0, 64));
			if (qzSuspect) {
				pads.push_back(std::clamp(basePad * 4, basePad * 3 + 1, 384));
				pads.push_back(std::clamp(basePad * 6, basePad * 4 + 1, 512));
			}
		}
	}
	std::sort(pads.begin(), pads.end());
	pads.erase(std::unique(pads.begin(), pads.end()), pads.end());

	ZXing::Barcode barcode;
	ZXing::Barcode bestInvalid;
	bool haveInvalid = false;
	int padUsed = 0;
	const char* binUsed = "local";
	int lastPadTried = -1;
	uint8_t binMask = 0;
	for (int pad : pads) {
		lastPadTried = pad;
		int paddedW = 0;
		int paddedH = 0;
		std::vector<uint8_t> padded = addWhiteBorder(base.data(), baseW, baseH, pad, paddedW, paddedH);
		auto image = ZXing::ImageView(padded.data(), paddedW, paddedH, ZXing::ImageFormat::Lum);

	ZXing::ReaderOptions opts;
		opts.setFormats(ZXing::BarcodeFormat::QRCode | ZXing::BarcodeFormat::MicroQRCode);
		opts.setTryHarder(tryHarder == JNI_TRUE);
		opts.setTryInvert(true);
		opts.setTryRotate(false);
		opts.setTryDownscale(tryHarder == JNI_TRUE);
		opts.setReturnErrors(true);
#ifdef ZXING_EXPERIMENTAL_API
		opts.setTryDenoise(true);
#endif

		binUsed = "local";
		barcode = ZXing::ReadBarcode(image, opts);
		barcode = retryAsQRCodeIfMicroInvalid(image, opts, barcode);
		binMask |= 1;
		if (barcode.format() != ZXing::BarcodeFormat::None && !barcode.isValid() && !haveInvalid) {
			bestInvalid = barcode;
			haveInvalid = true;
		}
		if (barcode.error() == ZXing::Error::Format)
			barcode = ZXing::Barcode();
		if (barcode.format() == ZXing::BarcodeFormat::None && tryHarder == JNI_TRUE) {
			opts.setBinarizer(ZXing::Binarizer::GlobalHistogram);
			binUsed = "global";
			barcode = ZXing::ReadBarcode(image, opts);
			barcode = retryAsQRCodeIfMicroInvalid(image, opts, barcode);
			binMask |= 2;
			if (barcode.format() != ZXing::BarcodeFormat::None && !barcode.isValid() && !haveInvalid) {
				bestInvalid = barcode;
				haveInvalid = true;
			}
			if (barcode.error() == ZXing::Error::Format)
				barcode = ZXing::Barcode();
		}
		if (barcode.format() == ZXing::BarcodeFormat::None && tryHarder == JNI_TRUE) {
			opts.setBinarizer(ZXing::Binarizer::FixedThreshold);
			binUsed = "fixed";
			barcode = ZXing::ReadBarcode(image, opts);
			barcode = retryAsQRCodeIfMicroInvalid(image, opts, barcode);
			binMask |= 4;
			if (barcode.format() != ZXing::BarcodeFormat::None && !barcode.isValid() && !haveInvalid) {
				bestInvalid = barcode;
				haveInvalid = true;
			}
			if (barcode.error() == ZXing::Error::Format)
				barcode = ZXing::Barcode();
		}

		if (barcode.format() != ZXing::BarcodeFormat::None) {
			padUsed = pad;
			break;
		}
	}
	if (barcode.format() == ZXing::BarcodeFormat::None && haveInvalid)
		barcode = bestInvalid;

	std::string out = "f=" + ToString(barcode.format());
	out += " v=" + std::to_string(barcode.isValid() ? 1 : 0);
	out += " e=" + (barcode.error() ? ToString(barcode.error()) : std::string("None"));
	out += " inv=" + std::to_string(barcode.isInverted() ? 1 : 0);
	out += " mir=" + std::to_string(barcode.isMirrored() ? 1 : 0);
	out += " o=" + std::to_string(barcode.orientation());
	out += " bin=" + std::string(binUsed);
	out += " bmask=" + std::to_string(static_cast<int>(binMask));
	out += " pad=" + std::to_string(padUsed);
	out += " lastpad=" + std::to_string(lastPadTried);
	out += " qz=" + std::to_string(margins.left) + "," + std::to_string(margins.top) + "," + std::to_string(margins.right) + "," +
		   std::to_string(margins.bottom);
	out += " pside=" + std::to_string(padL) + "," + std::to_string(padT) + "," + std::to_string(padR) + "," + std::to_string(padB);
	out += " tqz=" + std::to_string(targetQz);
	out += " bpad=" + std::to_string(basePad);
	out += " pads=";
	for (size_t i = 0; i < pads.size(); i++) {
		out += std::to_string(pads[i]);
		if (i + 1 < pads.size())
			out += ",";
	}
	out += " roi=" + formatRect(roi);
	out += " crop=" + std::to_string(cropW) + "x" + std::to_string(cropH);
	out += " rot=" + std::to_string(rotW) + "x" + std::to_string(rotH);
	out += " base=" + std::to_string(baseW) + "x" + std::to_string(baseH);
	out += " th=" + std::to_string((tryHarder == JNI_TRUE) ? 1 : 0);

	return env->NewStringUTF(out.c_str());
}
