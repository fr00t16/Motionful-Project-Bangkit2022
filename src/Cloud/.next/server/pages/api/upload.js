"use strict";
/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
(() => {
var exports = {};
exports.id = "pages/api/upload";
exports.ids = ["pages/api/upload"];
exports.modules = {

/***/ "multer":
/*!*************************!*\
  !*** external "multer" ***!
  \*************************/
/***/ ((module) => {

module.exports = require("multer");

/***/ }),

/***/ "next-connect":
/*!*******************************!*\
  !*** external "next-connect" ***!
  \*******************************/
/***/ ((module) => {

module.exports = import("next-connect");;

/***/ }),

/***/ "(api)/./pages/api/upload.js":
/*!*****************************!*\
  !*** ./pages/api/upload.js ***!
  \*****************************/
/***/ ((module, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.a(module, async (__webpack_handle_async_dependencies__, __webpack_async_result__) => { try {\n__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   \"config\": () => (/* binding */ config),\n/* harmony export */   \"default\": () => (__WEBPACK_DEFAULT_EXPORT__)\n/* harmony export */ });\n/* harmony import */ var next_connect__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! next-connect */ \"next-connect\");\n/* harmony import */ var multer__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! multer */ \"multer\");\n/* harmony import */ var multer__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(multer__WEBPACK_IMPORTED_MODULE_1__);\nvar __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([next_connect__WEBPACK_IMPORTED_MODULE_0__]);\nnext_connect__WEBPACK_IMPORTED_MODULE_0__ = (__webpack_async_dependencies__.then ? (await __webpack_async_dependencies__)() : __webpack_async_dependencies__)[0];\n\n\nconst upload = multer__WEBPACK_IMPORTED_MODULE_1___default()({\n    storage: multer__WEBPACK_IMPORTED_MODULE_1___default().diskStorage({\n        destination: \"./public/motion\",\n        filename: (req, file, cb)=>cb(null, file.originalname)\n    })\n});\nconst apiRoute = (0,next_connect__WEBPACK_IMPORTED_MODULE_0__[\"default\"])({\n    onError (error, req, res) {\n        res.status(501).json({\n            error: `Sorry something Happened! ${error.message}`\n        });\n    },\n    onNoMatch (req, res) {\n        res.status(405).json({\n            error: `Metode '${req.method}' Tidak di perbolehkan`\n        });\n    }\n});\napiRoute.use(upload.array(\"theFiles\"));\napiRoute.post((req, res)=>{\n    res.status(200).json({\n        data: \"success\"\n    });\n});\n/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (apiRoute);\nconst config = {\n    api: {\n        bodyParser: false\n    }\n};\n\n__webpack_async_result__();\n} catch(e) { __webpack_async_result__(e); } });//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKGFwaSkvLi9wYWdlcy9hcGkvdXBsb2FkLmpzLmpzIiwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7O0FBQXVDO0FBQ1g7QUFFNUIsTUFBTUUsTUFBTSxHQUFHRCw2Q0FBTSxDQUFDO0lBQ3BCRSxPQUFPLEVBQUVGLHlEQUFrQixDQUFDO1FBQzFCSSxXQUFXLEVBQUUsaUJBQWlCO1FBQzlCQyxRQUFRLEVBQUUsQ0FBQ0MsR0FBRyxFQUFFQyxJQUFJLEVBQUVDLEVBQUUsR0FBS0EsRUFBRSxDQUFDLElBQUksRUFBRUQsSUFBSSxDQUFDRSxZQUFZLENBQUM7S0FDekQsQ0FBQztDQUNILENBQUM7QUFFRixNQUFNQyxRQUFRLEdBQUdYLHdEQUFXLENBQUM7SUFDM0JZLE9BQU8sRUFBQ0MsS0FBSyxFQUFFTixHQUFHLEVBQUVPLEdBQUcsRUFBRTtRQUN2QkEsR0FBRyxDQUFDQyxNQUFNLENBQUMsR0FBRyxDQUFDLENBQUNDLElBQUksQ0FBQztZQUFFSCxLQUFLLEVBQUUsQ0FBQywwQkFBMEIsRUFBRUEsS0FBSyxDQUFDSSxPQUFPLENBQUMsQ0FBQztTQUFFLENBQUMsQ0FBQztLQUMvRTtJQUNEQyxTQUFTLEVBQUNYLEdBQUcsRUFBRU8sR0FBRyxFQUFFO1FBQ2xCQSxHQUFHLENBQUNDLE1BQU0sQ0FBQyxHQUFHLENBQUMsQ0FBQ0MsSUFBSSxDQUFDO1lBQUVILEtBQUssRUFBRSxDQUFDLFFBQVEsRUFBRU4sR0FBRyxDQUFDWSxNQUFNLENBQUMsc0JBQXNCLENBQUM7U0FBRSxDQUFDLENBQUM7S0FDaEY7Q0FDRixDQUFDO0FBRUZSLFFBQVEsQ0FBQ1MsR0FBRyxDQUFDbEIsTUFBTSxDQUFDbUIsS0FBSyxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUM7QUFFdkNWLFFBQVEsQ0FBQ1csSUFBSSxDQUFDLENBQUNmLEdBQUcsRUFBRU8sR0FBRyxHQUFLO0lBQ3hCQSxHQUFHLENBQUNDLE1BQU0sQ0FBQyxHQUFHLENBQUMsQ0FBQ0MsSUFBSSxDQUFDO1FBQUVPLElBQUksRUFBRSxTQUFTO0tBQUUsQ0FBQyxDQUFDO0NBQzdDLENBQUMsQ0FBQztBQUVILGlFQUFlWixRQUFRLEVBQUM7QUFFakIsTUFBTWEsTUFBTSxHQUFHO0lBQ3BCQyxHQUFHLEVBQUU7UUFDSEMsVUFBVSxFQUFFLEtBQUs7S0FDbEI7Q0FDRixDQUFDIiwic291cmNlcyI6WyJ3ZWJwYWNrOi8vbW90aW9uZnVsLWJhbmdraXQyMDIyLy4vcGFnZXMvYXBpL3VwbG9hZC5qcz81NTcyIl0sInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBuZXh0Q29ubmVjdCBmcm9tICduZXh0LWNvbm5lY3QnO1xuaW1wb3J0IG11bHRlciBmcm9tICdtdWx0ZXInO1xuXG5jb25zdCB1cGxvYWQgPSBtdWx0ZXIoe1xuICBzdG9yYWdlOiBtdWx0ZXIuZGlza1N0b3JhZ2Uoe1xuICAgIGRlc3RpbmF0aW9uOiAnLi9wdWJsaWMvbW90aW9uJyxcbiAgICBmaWxlbmFtZTogKHJlcSwgZmlsZSwgY2IpID0+IGNiKG51bGwsIGZpbGUub3JpZ2luYWxuYW1lKSxcbiAgfSksXG59KTtcblxuY29uc3QgYXBpUm91dGUgPSBuZXh0Q29ubmVjdCh7XG4gIG9uRXJyb3IoZXJyb3IsIHJlcSwgcmVzKSB7XG4gICAgcmVzLnN0YXR1cyg1MDEpLmpzb24oeyBlcnJvcjogYFNvcnJ5IHNvbWV0aGluZyBIYXBwZW5lZCEgJHtlcnJvci5tZXNzYWdlfWAgfSk7XG4gIH0sXG4gIG9uTm9NYXRjaChyZXEsIHJlcykge1xuICAgIHJlcy5zdGF0dXMoNDA1KS5qc29uKHsgZXJyb3I6IGBNZXRvZGUgJyR7cmVxLm1ldGhvZH0nIFRpZGFrIGRpIHBlcmJvbGVoa2FuYCB9KTtcbiAgfSxcbn0pO1xuXG5hcGlSb3V0ZS51c2UodXBsb2FkLmFycmF5KCd0aGVGaWxlcycpKTtcblxuYXBpUm91dGUucG9zdCgocmVxLCByZXMpID0+IHsgICAgXG4gICAgcmVzLnN0YXR1cygyMDApLmpzb24oeyBkYXRhOiAnc3VjY2VzcycgfSk7XG59KTtcblxuZXhwb3J0IGRlZmF1bHQgYXBpUm91dGU7XG5cbmV4cG9ydCBjb25zdCBjb25maWcgPSB7XG4gIGFwaToge1xuICAgIGJvZHlQYXJzZXI6IGZhbHNlLFxuICB9LFxufTsiXSwibmFtZXMiOlsibmV4dENvbm5lY3QiLCJtdWx0ZXIiLCJ1cGxvYWQiLCJzdG9yYWdlIiwiZGlza1N0b3JhZ2UiLCJkZXN0aW5hdGlvbiIsImZpbGVuYW1lIiwicmVxIiwiZmlsZSIsImNiIiwib3JpZ2luYWxuYW1lIiwiYXBpUm91dGUiLCJvbkVycm9yIiwiZXJyb3IiLCJyZXMiLCJzdGF0dXMiLCJqc29uIiwibWVzc2FnZSIsIm9uTm9NYXRjaCIsIm1ldGhvZCIsInVzZSIsImFycmF5IiwicG9zdCIsImRhdGEiLCJjb25maWciLCJhcGkiLCJib2R5UGFyc2VyIl0sInNvdXJjZVJvb3QiOiIifQ==\n//# sourceURL=webpack-internal:///(api)/./pages/api/upload.js\n");

/***/ })

};
;

// load runtime
var __webpack_require__ = require("../../webpack-api-runtime.js");
__webpack_require__.C(exports);
var __webpack_exec__ = (moduleId) => (__webpack_require__(__webpack_require__.s = moduleId))
var __webpack_exports__ = (__webpack_exec__("(api)/./pages/api/upload.js"));
module.exports = __webpack_exports__;

})();